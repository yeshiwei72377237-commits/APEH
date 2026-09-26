#!/usr/bin/env python3
"""APEH 的 ProteinMPNN Windows 封装接口。

功能：
1. 接收运行目录（单运行目录策略），自动发现 PDB 与推荐位置；
2. 检查输入文件与 ProteinMPNN 安装目录；
3. 处理 ProteinMPNN 官方脚本在 Windows 上的路径兼容问题；
4. 调用 ProteinMPNN 生成候选 FASTA；
5. 把候选转换成分数+突变列表（突变用 PDB 残基号标注）；
6. 去重并标记新引入 Pro/Cys 等风险；
7. 保存完整运行日志与链内序号→PDB 残基号映射。

运行目录约定：
    <dir>/<name>.pdb            输入结构（目录内唯一 *.pdb）
    <dir>/recommended_positions.csv  可选，配合 --from-recommendations
    <dir>/seqs/<name>.fa        ProteinMPNN 输出
    <dir>/candidates.csv        候选 + 突变列表
    <dir>/audited_candidates.csv 去重审核
    <dir>/resid_mapping.json    链内序号→PDB 残基号映射
    <dir>/proteinmpnn.log       完整日志
    <dir>/.work/、<dir>/constraints/ 中间 JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pdb_utils import parse_pdb_residues


MODEL_NAME = "v_48_020"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="proteinmpnn_wrapper.py",
        description="在 Windows 上调用 ProteinMPNN 生成候选蛋白质序列。",
    )

    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="运行目录：内含唯一 *.pdb，输出也写回该目录。",
    )

    parser.add_argument(
        "--proteinmpnn-dir",
        type=Path,
        default=None,
        help=(
            "官方 ProteinMPNN 仓库目录。缺省时自动定位到 "
            "本脚本所在目录下的 ProteinMPNN-main/。"
        ),
    )

    parser.add_argument(
        "--chains",
        default="",
        help='需要设计的链，例如 "A" 或 "A B"；不填写时设计全部链。',
    )

    parser.add_argument(
        "--design-positions",
        default="",
        help=(
            '仅允许设计的残基，例如 "A35 A68 A102"。'
            "位置是每条链氨基酸序列中从1开始的序号，不一定等于PDB残基编号。"
        ),
    )

    parser.add_argument(
        "--from-recommendations",
        action="store_true",
        help=(
            "从 <dir>/recommended_positions.csv 读取 position 列作为设计位置，"
            "替代 --design-positions（仍需先人工确认推荐结果）。"
        ),
    )

    parser.add_argument(
        "--num-sequences",
        type=int,
        default=20,
        help="生成的候选序列数量，默认20。",
    )

    parser.add_argument(
        "--sampling-temp",
        type=float,
        default=0.1,
        help="ProteinMPNN采样温度，默认0.1。",
    )

    parser.add_argument(
        "--omit-aas",
        default="",
        help=(
            '禁止ProteinMPNN在设计位置使用的氨基酸单字母，例如 "PC"。'
            "默认不额外禁止；未知残基X始终被禁止。"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=37,
        help="随机种子，默认37。",
    )

    return parser


def discover_pdb(run_dir: Path) -> Path:
    """在运行目录中发现唯一的 PDB 输入。"""
    pdb_files = sorted(run_dir.glob("*.pdb"))
    if len(pdb_files) != 1:
        raise ValueError(
            f"运行目录必须恰好包含一个 .pdb 文件，当前找到 {len(pdb_files)} 个："
            f"{run_dir}"
        )
    return pdb_files[0]


def read_recommended_positions(run_dir: Path) -> str:
    """读取推荐器输出的 position 列，拼成 design-positions 字符串。"""
    csv_path = run_dir / "recommended_positions.csv"
    if not csv_path.is_file():
        raise ValueError(f"找不到推荐表：{csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "position" not in reader.fieldnames:
            raise ValueError(f"推荐表缺少 position 列：{csv_path}")
        positions = [
            row["position"].strip()
            for row in reader
            if row.get("position", "").strip()
        ]
    if not positions:
        raise ValueError(f"推荐表中没有 position 数据：{csv_path}")
    return " ".join(positions)


def parse_design_positions(
    value: str,
    chains_argument: str,
) -> dict[str, list[int]]:
    """解析A35一类位置，并按链返回去重、排序后的序号。

    未填写 --chains 时，链顺序由位置令牌首次出现的顺序推导。
    """

    if not value.strip():
        return {}

    requested_chains = chains_argument.split()
    if not requested_chains:
        requested_chains = []
        for token in value.split():
            match = re.fullmatch(r"([A-Za-z0-9])(\d+)", token)
            if not match:
                raise ValueError(
                    f"--design-positions格式错误："
                    f"{token}；请使用A35 A68 A102这种格式"
                )
            chain = match.group(1)
            if chain not in requested_chains:
                requested_chains.append(chain)

    if len(set(requested_chains)) != len(requested_chains):
        raise ValueError("--chains中不能出现重复链名")

    for chain in requested_chains:
        if not re.fullmatch(r"[A-Za-z0-9]", chain):
            raise ValueError(
                "当前版本的链名必须是一个字母或数字："
                f"{chain}"
            )

    positions_by_chain: dict[str, set[int]] = {
        chain: set() for chain in requested_chains
    }

    for token in value.split():
        match = re.fullmatch(r"([A-Za-z0-9])(\d+)", token)
        if not match:
            raise ValueError(
                "--design-positions格式错误："
                f"{token}；请使用A35 A68 A102这种格式"
            )

        chain, position_text = match.groups()
        if chain not in positions_by_chain:
            raise ValueError(
                f"设计位置{token}属于链{chain}，"
                f"但--chains中没有该链"
            )

        position = int(position_text)
        if position < 1:
            raise ValueError(f"设计位置必须从1开始：{token}")
        positions_by_chain[chain].add(position)

    missing_chains = [
        chain
        for chain, positions in positions_by_chain.items()
        if not positions
    ]
    if missing_chains:
        raise ValueError(
            "以下设计链没有指定任何可变位置："
            + " ".join(missing_chains)
        )

    return {
        chain: sorted(positions)
        for chain, positions in positions_by_chain.items()
    }


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """检查输入参数是否有效。"""

    args.dir = args.dir.expanduser().resolve()
    if not args.dir.is_dir():
        parser.error(f"运行目录不存在：{args.dir}")
    args.pdb_path = discover_pdb(args.dir)
    args.name = args.pdb_path.stem

    if args.proteinmpnn_dir:
        args.proteinmpnn_dir = args.proteinmpnn_dir.expanduser().resolve()
    else:
        args.proteinmpnn_dir = (
            Path(__file__).resolve().parent / "ProteinMPNN-main"
        ).resolve()

    if not args.proteinmpnn_dir.is_dir():
        parser.error(
            f"ProteinMPNN目录不存在：{args.proteinmpnn_dir}"
        )

    mpnn_script = args.proteinmpnn_dir / "protein_mpnn_run.py"
    if not mpnn_script.is_file():
        parser.error(
            f"找不到ProteinMPNN主程序：{mpnn_script}"
        )

    model_file = (
        args.proteinmpnn_dir
        / "vanilla_model_weights"
        / f"{MODEL_NAME}.pt"
    )
    if not model_file.is_file():
        parser.error(
            f"找不到ProteinMPNN模型权重：{model_file}"
        )

    if args.num_sequences <= 0:
        parser.error("--num-sequences必须大于0")

    if args.sampling_temp <= 0:
        parser.error("--sampling-temp必须大于0")

    args.omit_aas = args.omit_aas.strip().upper()
    invalid_omissions = sorted(set(args.omit_aas) - VALID_AMINO_ACIDS)
    if invalid_omissions:
        parser.error(
            "--omit-aas包含无效氨基酸字母："
            + " ".join(invalid_omissions)
        )

    if args.from_recommendations and args.design_positions.strip():
        parser.error("--from-recommendations 与 --design-positions 不能同时使用")

    if args.from_recommendations:
        try:
            design_value = read_recommended_positions(args.dir)
        except ValueError as error:
            parser.error(str(error))
    else:
        design_value = args.design_positions

    try:
        args.design_position_map = parse_design_positions(
            design_value,
            args.chains,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.design_position_map:
        helper_scripts = [
            "parse_multiple_chains.py",
            "assign_fixed_chains.py",
            "make_fixed_positions_dict.py",
        ]
        for helper_name in helper_scripts:
            helper_path = (
                args.proteinmpnn_dir
                / "helper_scripts"
                / helper_name
            )
            if not helper_path.is_file():
                parser.error(
                    f"找不到ProteinMPNN辅助脚本：{helper_path}"
                )


def print_config(args: argparse.Namespace) -> None:
    """显示本次运行配置。"""

    print("=" * 60)
    print("APEH ProteinMPNN Windows Wrapper")
    print("=" * 60)
    print(f"运行目录：      {args.dir}")
    print(f"输入PDB：       {args.pdb_path}")
    print(f"ProteinMPNN：   {args.proteinmpnn_dir}")
    print(f"设计链：        {args.chains or '全部链'}")
    if args.design_position_map:
        design_text = " ".join(
            f"{chain}{pos}"
            for chain, positions in args.design_position_map.items()
            for pos in positions
        )
    else:
        design_text = "整条设计链"
    print(f"设计位置：      {design_text}")
    print(f"候选数量：      {args.num_sequences}")
    print(f"采样温度：      {args.sampling_temp}")
    print(f"禁止氨基酸：    {args.omit_aas or '无（X始终禁止）'}")
    print(f"随机种子：      {args.seed}")
    print(f"模型：          {MODEL_NAME}")
    print("=" * 60)


def make_safe_stem(original_stem: str) -> str:
    """把文件名转换成适合 ProteinMPNN 输出的安全名称。"""

    safe_stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in original_stem
    )
    return safe_stem or "input_structure"


def decode_output(raw: bytes | None) -> str:
    """兼容 UTF-8 和中文 Windows编码。"""

    if not raw:
        return ""

    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def write_run_log(
    log_path: Path,
    sections: list[str],
) -> None:
    """把截至当前的全部子命令日志写入文件。"""

    log_path.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )


def run_logged_command(
    command: list[str],
    cwd: Path,
    label: str,
    log_path: Path,
    log_sections: list[str],
) -> None:
    """运行一个子命令、保存输出，并在失败时给出可读错误。"""

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    stdout_text = decode_output(result.stdout)
    stderr_text = decode_output(result.stderr)
    command_text = subprocess.list2cmdline(command)

    log_sections.append(
        f"Step: {label}\n"
        f"Command:\n{command_text}\n\n"
        f"Return code:\n{result.returncode}\n\n"
        f"STDOUT:\n{stdout_text}\n\n"
        f"STDERR:\n{stderr_text}"
    )
    write_run_log(log_path, log_sections)

    if stdout_text.strip():
        print(stdout_text.strip())

    if result.returncode != 0:
        error_tail = (
            stderr_text.strip()
            or stdout_text.strip()
            or "没有错误输出"
        )
        raise RuntimeError(
            f"{label}失败。\n"
            f"退出码：{result.returncode}\n"
            f"日志：{log_path}\n"
            f"错误信息：\n{error_tail[-2000:]}"
        )


def sanitize_parsed_jsonl(
    source_path: Path,
    destination_path: Path,
    safe_stem: str,
) -> None:
    """清除Windows辅助脚本写入结构名称的目录前缀。"""

    records: list[dict[str, object]] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["name"] = safe_stem
        records.append(record)

    if not records:
        raise RuntimeError(
            f"结构解析结果为空：{source_path}"
        )
    if len(records) != 1:
        raise RuntimeError(
            "本接口一次只处理一个PDB，但解析结果中发现"
            f"{len(records)}个结构"
        )

    destination_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def validate_positions_against_structure(
    parsed_path: Path,
    positions_by_chain: dict[str, list[int]],
) -> None:
    """确认指定链存在，且链内序号没有超出氨基酸序列长度。"""

    record = json.loads(
        parsed_path.read_text(encoding="utf-8").splitlines()[0]
    )

    for chain, positions in positions_by_chain.items():
        sequence_key = f"seq_chain_{chain}"
        sequence = record.get(sequence_key)
        if not isinstance(sequence, str):
            available_chains = sorted(
                key.removeprefix("seq_chain_")
                for key in record
                if key.startswith("seq_chain_")
            )
            raise RuntimeError(
                f"PDB中没有找到链{chain}；"
                f"可用链：{' '.join(available_chains) or '无'}"
            )

        too_large = [
            position
            for position in positions
            if position > len(sequence)
        ]
        if too_large:
            raise RuntimeError(
                f"链{chain}长度为{len(sequence)}，"
                "以下链内序号超出范围："
                + " ".join(map(str, too_large))
            )


def build_position_constraints(
    args: argparse.Namespace,
    work_dir: Path,
    safe_stem: str,
    log_path: Path,
    log_sections: list[str],
) -> tuple[Path, Path, Path]:
    """调用官方辅助脚本，生成指定位置设计所需的JSONL。"""

    helpers_dir = args.proteinmpnn_dir / "helper_scripts"
    parsed_raw = work_dir / "parsed_raw.jsonl"
    parsed_path = work_dir / "parsed_pdbs.jsonl"
    assigned_path = work_dir / "assigned_pdbs.jsonl"
    fixed_path = work_dir / "fixed_pdbs.jsonl"

    run_logged_command(
        [
            sys.executable,
            str(helpers_dir / "parse_multiple_chains.py"),
            "--input_path",
            "./",
            "--output_path",
            str(parsed_raw),
        ],
        work_dir,
        "解析PDB结构",
        log_path,
        log_sections,
    )
    sanitize_parsed_jsonl(parsed_raw, parsed_path, safe_stem)
    validate_positions_against_structure(
        parsed_path,
        args.design_position_map,
    )

    chain_list = " ".join(args.design_position_map)
    run_logged_command(
        [
            sys.executable,
            str(helpers_dir / "assign_fixed_chains.py"),
            "--input_path",
            str(parsed_path),
            "--output_path",
            str(assigned_path),
            "--chain_list",
            chain_list,
        ],
        work_dir,
        "指定设计链",
        log_path,
        log_sections,
    )

    position_list = ", ".join(
        " ".join(map(str, args.design_position_map[chain]))
        for chain in args.design_position_map
    )
    run_logged_command(
        [
            sys.executable,
            str(helpers_dir / "make_fixed_positions_dict.py"),
            "--input_path",
            str(parsed_path),
            "--output_path",
            str(fixed_path),
            "--chain_list",
            chain_list,
            "--position_list",
            position_list,
            "--specify_non_fixed",
        ],
        work_dir,
        "生成固定位置约束",
        log_path,
        log_sections,
    )

    constraints_dir = args.dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    for path in (parsed_path, assigned_path, fixed_path):
        shutil.copy2(path, constraints_dir / path.name)

    return parsed_path, assigned_path, fixed_path


def run_proteinmpnn(args: argparse.Namespace) -> Path:
    """调用官方 ProteinMPNN并返回生成的 FASTA路径。"""

    work_dir = args.dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.dir / "proteinmpnn.log"
    log_sections: list[str] = []

    safe_stem = make_safe_stem(args.name)
    temporary_pdb = work_dir / f"{safe_stem}.pdb"
    shutil.copy2(args.pdb_path, temporary_pdb)

    mpnn_script = args.proteinmpnn_dir / "protein_mpnn_run.py"
    weights_dir = args.proteinmpnn_dir / "vanilla_model_weights"

    command = [
        sys.executable,
        str(mpnn_script),
        "--path_to_model_weights",
        str(weights_dir),
        "--model_name",
        MODEL_NAME,
        "--out_folder",
        str(args.dir),
        "--num_seq_per_target",
        str(args.num_sequences),
        "--sampling_temp",
        f"{args.sampling_temp:g}",
        "--seed",
        str(args.seed),
        "--batch_size",
        "1",
        # 未知残基 X 始终禁止，避免 --omit-aas 覆盖官方默认后放开 X。
        "--omit_AAs",
        "X" + args.omit_aas,
    ]

    if args.design_position_map:
        parsed_path, assigned_path, fixed_path = (
            build_position_constraints(
                args,
                work_dir,
                safe_stem,
                log_path,
                log_sections,
            )
        )
        command.extend([
            "--jsonl_path",
            str(parsed_path),
            "--chain_id_jsonl",
            str(assigned_path),
            "--fixed_positions_jsonl",
            str(fixed_path),
        ])
    else:
        command.extend([
            "--pdb_path",
            temporary_pdb.name,
        ])

    if args.chains.strip() and not args.design_position_map:
        command.extend([
            "--pdb_path_chains",
            args.chains.strip(),
        ])

    print()
    print("正在运行ProteinMPNN……")

    run_logged_command(
        command,
        work_dir,
        "运行ProteinMPNN",
        log_path,
        log_sections,
    )

    fasta_path = args.dir / "seqs" / f"{safe_stem}.fa"

    if not fasta_path.is_file():
        raise RuntimeError(
            "ProteinMPNN命令已结束，但没有找到预期的FASTA文件："
            f"{fasta_path}\n"
            f"请检查日志：{log_path}"
        )

    print()
    print("ProteinMPNN运行成功。")
    print(f"FASTA结果：{fasta_path}")
    print(f"运行日志： {log_path}")

    return fasta_path


def read_fasta_records(fasta_path: Path) -> list[tuple[str, str]]:
    """读取FASTA并返回“标题、序列”记录。"""

    records: list[tuple[str, str]] = []
    current_header: str | None = None
    sequence_parts: list[str] = []

    for raw_line in fasta_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None:
                records.append(
                    (current_header, "".join(sequence_parts))
                )

            current_header = line[1:]
            sequence_parts = []
        else:
            sequence_parts.append(line)

    if current_header is not None:
        records.append(
            (current_header, "".join(sequence_parts))
        )

    return records


def get_header_value(header: str, key: str) -> str:
    """从ProteinMPNN的FASTA标题中读取一个字段。"""

    match = re.search(
        rf"(?:^|,\s*){re.escape(key)}=([^,]+)",
        header,
    )
    return match.group(1).strip() if match else ""


def get_designed_chains(header: str) -> list[str]:
    """从 FASTA 标题的 designed_chains=[...] 解析设计链顺序。"""
    match = re.search(r"designed_chains=\[([^\]]*)\]", header)
    if not match:
        return []
    return re.findall(r"[A-Za-z0-9]+", match.group(1))


def build_resid_map(
    pdb_path: Path,
    chain_labels: list[str],
) -> dict[str, list[int]]:
    """为每条设计链建立链内序号→PDB残基号的映射。"""
    resid_map: dict[str, list[int]] = {}
    for chain in chain_labels:
        try:
            residues = parse_pdb_residues(pdb_path, chain)
            resid_map[chain] = [residue.pdb_resid for residue in residues]
        except ValueError:
            resid_map[chain] = []
    return resid_map


def write_resid_mapping(
    resid_map: dict[str, list[int]],
    chain_labels: list[str],
    path: Path,
) -> None:
    """写链内序号→PDB残基号映射，供下游溯源。"""
    payload = {
        "chains": chain_labels,
        "mapping": {
            chain: resid_map.get(chain, [])
            for chain in chain_labels
        },
        "note": "mapping 列表中下标 i 对应链内1起始序号 i+1；"
                "mutations 中的数字为 PDB 残基号。",
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compare_sequences(
    reference_sequence: str,
    candidate_sequence: str,
    chain_labels: list[str],
    resid_map: dict[str, list[int]],
) -> str:
    """比较参考序列与候选序列，生成突变列表。

    突变标签使用 PDB 残基号（resid）；若某链缺少映射则回退为
    链内1起始序号。
    """

    reference_chains = reference_sequence.split("/")
    candidate_chains = candidate_sequence.split("/")

    if len(reference_chains) != len(candidate_chains):
        return "CHAIN_COUNT_MISMATCH"

    mutations: list[str] = []

    for chain_label, reference, candidate in zip(
        chain_labels,
        reference_chains,
        candidate_chains,
    ):
        if len(reference) != len(candidate):
            mutations.append(
                f"{chain_label}:LENGTH_MISMATCH"
            )
            continue

        pdb_resids = resid_map.get(chain_label, [])
        for position, (old_aa, new_aa) in enumerate(
            zip(reference, candidate),
            start=1,
        ):
            if old_aa == new_aa:
                continue

            if old_aa == "X" or new_aa == "X":
                continue

            resid = (
                pdb_resids[position - 1]
                if position - 1 < len(pdb_resids)
                else position
            )
            mutations.append(
                f"{chain_label}:{old_aa}{resid}{new_aa}"
            )

    return ";".join(mutations)


def write_candidates_csv(
    fasta_path: Path,
    csv_path: Path,
    chain_labels: list[str],
    resid_map: dict[str, list[int]],
) -> int:
    """把ProteinMPNN FASTA转换成候选CSV。"""

    records = read_fasta_records(fasta_path)

    if len(records) < 2:
        raise RuntimeError(
            f"FASTA中没有找到候选序列：{fasta_path}"
        )

    reference_header, reference_sequence = records[0]
    candidate_records = records[1:]

    fieldnames = [
        "candidate_id",
        "sample",
        "sampling_temp",
        "mpnn_score",
        "global_score",
        "seq_recovery",
        "mutations",
        "sequence",
        "reference_header",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for candidate_id, (
            header,
            sequence,
        ) in enumerate(candidate_records, start=1):
            writer.writerow({
                "candidate_id": candidate_id,
                "sample": get_header_value(
                    header,
                    "sample",
                ),
                "sampling_temp": get_header_value(
                    header,
                    "T",
                ),
                "mpnn_score": get_header_value(
                    header,
                    "score",
                ),
                "global_score": get_header_value(
                    header,
                    "global_score",
                ),
                "seq_recovery": get_header_value(
                    header,
                    "seq_recovery",
                ),
                "mutations": compare_sequences(
                    reference_sequence,
                    sequence,
                    chain_labels,
                    resid_map,
                ),
                "sequence": sequence,
                "reference_header": reference_header,
            })

    return len(candidate_records)


def audit_candidates_csv(csv_path: Path, audited_path: Path) -> tuple[int, int]:
    """候选去重并标记新引入Pro/Cys等需复核风险。"""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)

    def score_value(row: dict[str, str]) -> float:
        try:
            return float(row.get("mpnn_score", ""))
        except (TypeError, ValueError):
            return float("inf")

    unique_rows: list[dict[str, object]] = []
    for sequence, duplicates in grouped.items():
        best = min(duplicates, key=score_value)
        mutations = [item for item in best.get("mutations", "").split(";") if item]
        introduced = []
        for mutation in mutations:
            match = re.fullmatch(r"[^:]+:[A-Z]\d+([A-Z])", mutation)
            if match:
                introduced.append(match.group(1))

        risk_flags: list[str] = []
        if "P" in introduced:
            risk_flags.append("NEW_PROLINE")
        if "C" in introduced:
            risk_flags.append("NEW_CYSTEINE")
        if any("MISMATCH" in mutation for mutation in mutations):
            risk_flags.append("SEQUENCE_MISMATCH")

        status = "HOLD_FOR_STRUCTURE_REVIEW" if risk_flags else "PASS_TO_STRUCTURE_MODEL"
        unique_rows.append({
            "best_mpnn_score": best.get("mpnn_score", ""),
            "source_candidate_ids": ";".join(row["candidate_id"] for row in duplicates),
            "duplicate_count": len(duplicates),
            "best_sample": best.get("sample", ""),
            "global_score": best.get("global_score", ""),
            "seq_recovery": best.get("seq_recovery", ""),
            "mutation_count": len(mutations),
            "mutations": best.get("mutations", ""),
            "risk_flags": ";".join(risk_flags),
            "review_status": status,
            "review_note": (
                "ProteinMPNN分数反映序列-骨架相容性，不代表热稳定性或酶活；"
                "需由后续结构模型及功能分析复核"
            ),
            "sequence": sequence,
        })

    unique_rows.sort(key=lambda row: float(row["best_mpnn_score"]) if row["best_mpnn_score"] else float("inf"))
    fieldnames = [
        "unique_candidate_id", "source_candidate_ids", "duplicate_count",
        "best_sample", "best_mpnn_score", "global_score", "seq_recovery",
        "mutation_count", "mutations", "risk_flags", "review_status",
        "review_note", "sequence",
    ]
    with audited_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for unique_id, row in enumerate(unique_rows, start=1):
            writer.writerow({"unique_candidate_id": unique_id, **row})

    passed = sum(row["review_status"] == "PASS_TO_STRUCTURE_MODEL" for row in unique_rows)
    return len(unique_rows), passed


def main() -> None:
    """程序入口。"""

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    print_config(args)

    try:
        fasta_path = run_proteinmpnn(args)

        records = read_fasta_records(fasta_path)
        if not records:
            raise RuntimeError(f"FASTA为空：{fasta_path}")
        chain_labels = get_designed_chains(records[0][0])
        if not chain_labels:
            chain_count = records[0][1].count("/") + 1
            chain_labels = [
                chr(ord("A") + index)
                for index in range(chain_count)
            ]

        resid_map = build_resid_map(args.pdb_path, chain_labels)
        write_resid_mapping(
            resid_map,
            chain_labels,
            args.dir / "resid_mapping.json",
        )

        csv_path = args.dir / "candidates.csv"
        candidate_count = write_candidates_csv(
            fasta_path,
            csv_path,
            chain_labels,
            resid_map,
        )

        audited_path = args.dir / "audited_candidates.csv"
        unique_count, passed_count = audit_candidates_csv(
            csv_path,
            audited_path,
        )

        print(f"候选CSV：  {csv_path}")
        print(f"候选数量： {candidate_count}")
        print(f"审核CSV：  {audited_path}")
        print(f"唯一序列： {unique_count}")
        print(f"可进入结构模型：{passed_count}")
        print(f"残基映射： {args.dir / 'resid_mapping.json'}")
    except RuntimeError as error:
        print()
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
