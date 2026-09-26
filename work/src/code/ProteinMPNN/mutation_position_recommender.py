#!/usr/bin/env python3
"""根据GROMACS逐残基RMSF和SASA生成候选突变位置建议。

本工具只做结构动力学层面的辅助排序，不判断催化、底物结合或序列保守性。
推荐结果必须经过人工确认后，才能交给ProteinMPNN。

运行目录约定（单运行目录策略）：
    目录内需包含唯一 *.pdb、rmsf.csv、residue_sasa.csv；
    输出 position_scores.csv 与 recommended_positions.csv 到同一目录。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from pdb_utils import MAX_ASA_A2, detect_first_chain, parse_pdb_residues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据GROMACS逐残基RMSF和SASA推荐候选突变位置",
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="运行目录：内含唯一 *.pdb、rmsf.csv、residue_sasa.csv")
    parser.add_argument("--chain", default="",
                        help="设计链；缺省时自动探测第一个含CA的链")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--exclude-positions", default="")
    parser.add_argument("--min-separation", type=int, default=3)
    parser.add_argument("--terminal-buffer", type=int, default=5)
    return parser


def discover_inputs(run_dir: Path) -> tuple[Path, Path, Path]:
    """在运行目录中发现 PDB 与 GROMACS 分析文件。"""
    pdb_files = sorted(run_dir.glob("*.pdb"))
    if len(pdb_files) != 1:
        raise ValueError(
            f"运行目录必须恰好包含一个 .pdb 文件，当前找到 {len(pdb_files)} 个："
            f"{run_dir}"
        )
    pdb_path = pdb_files[0]
    rmsf_path = run_dir / "rmsf.csv"
    residue_sasa_path = run_dir / "residue_sasa.csv"
    for required in (rmsf_path, residue_sasa_path):
        if not required.is_file():
            raise ValueError(f"运行目录缺少分析文件：{required}")
    return pdb_path, rmsf_path, residue_sasa_path


def read_numeric_csv(path: Path, key_name: str,
                     value_name: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key_name not in reader.fieldnames:
            raise ValueError(f"{path}缺少列：{key_name}")
        if value_name not in reader.fieldnames:
            raise ValueError(f"{path}缺少列：{value_name}")
        for row in reader:
            try:
                values[int(float(row[key_name]))] = float(row[value_name])
            except (TypeError, ValueError):
                continue
    if not values:
        raise ValueError(f"没有从{path}读到有效数据")
    return values


def percentile_ranks(values: list[float]) -> list[float]:
    """返回0~1百分位；相同值使用平均名次。"""
    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranks[index] = average_rank / (len(values) - 1)
        start = end
    return ranks


def parse_exclusions(value: str, chain: str) -> set[str]:
    exclusions: set[str] = set()
    for token in value.split():
        if not re.fullmatch(r"[A-Za-z0-9]\d+", token):
            raise ValueError(f"排除位置格式错误：{token}")
        if token[0] != chain:
            raise ValueError(f"排除位置{token}不属于链{chain}")
        exclusions.add(token)
    return exclusions


def terminal_histag_positions(residues) -> set[str]:
    """识别C端连续至少4个His组成的标签。"""
    count = 0
    for residue in reversed(residues):
        if residue.amino_acid != "H":
            break
        count += 1
    if count < 4:
        return set()
    return {residue.position for residue in residues[-count:]}


def align_values_to_residues(
    values: dict[int, float],
    residues,
    label: str,
) -> list[float]:
    """兼容PDB残基编号和1起始链内序号两种GROMACS输出。"""
    if all(residue.pdb_resid in values for residue in residues):
        return [values[residue.pdb_resid] for residue in residues]
    if all(residue.sequence_position in values for residue in residues):
        return [values[residue.sequence_position] for residue in residues]

    missing_pdb = [
        str(residue.pdb_resid)
        for residue in residues
        if residue.pdb_resid not in values
    ]
    raise ValueError(
        f"{label}与PDB残基无法完整对应；既不是PDB编号，也不是1起始链内序号。"
        "缺少的PDB编号示例：" + " ".join(missing_pdb[:20])
    )


def recommend_positions(
    pdb_path: Path,
    rmsf_path: Path,
    residue_sasa_path: Path,
    output_dir: Path,
    chain: str = "A",
    top_k: int = 10,
    exclude_positions: str = "",
    min_separation: int = 3,
    terminal_buffer: int = 5,
) -> tuple[Path, Path, list[dict[str, object]]]:
    if not re.fullmatch(r"[A-Za-z0-9]", chain):
        raise ValueError("链名必须是一个字母或数字")
    if top_k <= 0 or min_separation < 0 or terminal_buffer < 0:
        raise ValueError("top-k必须大于0，间隔和末端缓冲不能小于0")

    residues = parse_pdb_residues(pdb_path, chain)
    rmsf = read_numeric_csv(rmsf_path, "resid", "rmsf_nm")
    sasa = read_numeric_csv(residue_sasa_path, "resid", "sasa_nm2")
    manual_exclusions = parse_exclusions(exclude_positions, chain)
    histag = terminal_histag_positions(residues)

    rmsf_values = align_values_to_residues(rmsf, residues, "RMSF")
    sasa_values = align_values_to_residues(sasa, residues, "逐残基SASA")

    ranks = percentile_ranks(rmsf_values)
    rows: list[dict[str, object]] = []
    residue_count = len(residues)

    for residue, rmsf_nm, sasa_nm2, rmsf_percentile in zip(
        residues, rmsf_values, sasa_values, ranks
    ):
        max_asa_nm2 = MAX_ASA_A2.get(residue.amino_acid, 0) / 100
        relative_sasa = sasa_nm2 / max_asa_nm2 if max_asa_nm2 else 0.0
        exposure_score = min(max(relative_sasa, 0.0), 1.0)
        caution: list[str] = []
        excluded_reason = ""

        if residue.position in manual_exclusions:
            excluded_reason = "使用者指定排除"
        elif residue.position in histag:
            excluded_reason = "C端His-tag"
        elif residue.sequence_position <= terminal_buffer or (
            residue.sequence_position > residue_count - terminal_buffer
        ):
            excluded_reason = "靠近蛋白末端"
        elif residue.amino_acid == "X":
            excluded_reason = "非标准或未知残基"

        penalty = 0.0
        if residue.amino_acid in {"G", "P"}:
            caution.append("Gly/Pro可能影响局部骨架")
            penalty += 0.08
        if residue.amino_acid == "C":
            caution.append("Cys需确认是否参与二硫键")
            penalty += 0.12
        caution.append("需人工确认非催化、结合及保守位点")

        score = max(
            0.0,
            0.65 * rmsf_percentile + 0.35 * exposure_score - penalty,
        )
        reason = (
            f"RMSF位于第{rmsf_percentile * 100:.0f}百分位；"
            f"相对SASA约{relative_sasa:.2f}"
        )
        rows.append({
            "position": residue.position,
            "chain": residue.chain,
            "sequence_position": residue.sequence_position,
            "pdb_resid": residue.pdb_resid,
            "residue": residue.amino_acid,
            "residue_name": residue.residue_name,
            "rmsf_nm": rmsf_nm,
            "rmsf_percentile": rmsf_percentile,
            "sasa_nm2": sasa_nm2,
            "relative_sasa": relative_sasa,
            "recommendation_score": score,
            "excluded_reason": excluded_reason,
            "recommendation_reason": reason,
            "caution": "；".join(caution),
            "recommended_rank": "",
        })

    eligible = sorted(
        (row for row in rows if not row["excluded_reason"]),
        key=lambda row: float(row["recommendation_score"]),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    for row in eligible:
        position = int(row["sequence_position"])
        if any(
            abs(position - int(chosen["sequence_position"])) < min_separation
            for chosen in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= top_k:
            break

    for rank, row in enumerate(selected, start=1):
        row["recommended_rank"] = rank

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "position_scores.csv"
    recommended_path = output_dir / "recommended_positions.csv"
    fieldnames = list(rows[0])

    def write_csv(path: Path, data: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                formatted = dict(row)
                for key in ("rmsf_nm", "rmsf_percentile", "sasa_nm2",
                            "relative_sasa", "recommendation_score"):
                    formatted[key] = f"{float(formatted[key]):.6g}"
                writer.writerow(formatted)

    write_csv(all_path, rows)
    write_csv(recommended_path, selected)
    return recommended_path, all_path, selected


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dir = args.dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"运行目录不存在：{run_dir}")
        pdb_path, rmsf_path, residue_sasa_path = discover_inputs(run_dir)
        chain = args.chain or detect_first_chain(pdb_path)
        recommended_path, all_path, selected = recommend_positions(
            pdb_path=pdb_path,
            rmsf_path=rmsf_path,
            residue_sasa_path=residue_sasa_path,
            output_dir=run_dir,
            chain=chain,
            top_k=args.top_k,
            exclude_positions=args.exclude_positions,
            min_separation=args.min_separation,
            terminal_buffer=args.terminal_buffer,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print("\nGROMACS候选突变位置建议（必须人工确认）")
    print("=" * 72)
    for row in selected:
        print(
            f"{int(row['recommended_rank']):>2}. {row['position']:<6} "
            f"{row['residue']}  RMSF={float(row['rmsf_nm']):.3f} nm  "
            f"相对SASA={float(row['relative_sasa']):.2f}  "
            f"评分={float(row['recommendation_score']):.3f}"
        )
    print("=" * 72)
    print(f"推荐表：{recommended_path}")
    print(f"全部位置评分：{all_path}")
    print("提示：请排除催化、底物结合和高度保守位点后再交给ProteinMPNN。")


if __name__ == "__main__":
    main()
