#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 all_aaindex_nested_cv_auto6 冻结模型给 ProteinMPNN 候选打分。

做法：
    候选序列（356 aa，2UWF 构建体）
      → 按全局比对映射到 S7 360 aa 坐标（构建体未覆盖的位点保留 S7 野生型残基）
      → 按训练时同一份 AAindex 代码清单编码（553 个记录 × 360 位点 + AAC + DPC）
      → 套用冻结的折内流程（填补→零方差→预筛→标准化→选列）→ 模型预测活性

用法：
    python score.py --dir <ProteinMPNN 运行目录> [--artifacts artifacts] [--chain A]

产物：
    <dir>/scored_candidates.csv
        Mutant(S7/PDB 编号) / Mutant_seqpos(构建体编号) / Sequence / Fitness_prediction /
        mpnn_score / source_candidate_id / review_status / mutations_pdb / score_invalid /
        seq_len / model_name
    score_invalid: 0=有效, 1=长度或映射失败/打分异常, 2=审核门控拦截
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from feature_encoder import encode_with_codes  # noqa: E402
from map_sequence import (  # noqa: E402
    build_alignment_map,
    construct_to_s7,
    wt_construct_356,
)
from nested_cv import AA_SET, apply_fold  # noqa: E402
from safety_gate import is_blocked, normalize  # noqa: E402


def find_candidates(directory: Path, explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = directory / path
        if not path.is_file():
            raise SystemExit(f"找不到候选文件：{path}")
        return path
    for name in ("audited_candidates.csv", "candidates.csv"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{directory} 下找不到 audited_candidates.csv 或 candidates.csv")


def load_bundle(artifacts_dir: Path) -> dict:
    model_path = artifacts_dir / "final_pipeline.pkl"
    if not model_path.is_file():
        raise SystemExit(
            f"缺少模型产物：{model_path}；请先运行 train_final.py 冻结模型"
        )
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def normalize_candidate_columns(table: pd.DataFrame) -> pd.DataFrame:
    """统一候选表列名。

    audited_candidates.csv 用 best_mpnn_score / unique_candidate_id，
    candidates.csv 用 mpnn_score / candidate_id，需要归一到同一套列名，
    否则 mpnn_score / source_candidate_id 会被写成空值。
    （与 src/code/fitness_model/score.py 的 load_candidates 保持一致。）
    """
    rename = {}
    for source, target in (
        ("best_mpnn_score", "mpnn_score"),
        ("mpnn_score", "mpnn_score"),
        ("unique_candidate_id", "source_candidate_id"),
        ("candidate_id", "source_candidate_id"),
        ("best_sample", "sample"),
    ):
        if source in table.columns and target not in table.columns:
            rename[source] = target
    return table.rename(columns=rename) if rename else table


def describe_substitutions(candidate: str, reference: str, mapping: dict) -> tuple:
    """返回 (构建体编号标签, S7/PDB 编号标签)。"""
    seqpos, s7pos = [], []
    for index, (actual, expected) in enumerate(zip(candidate, reference)):
        if actual == expected:
            continue
        seqpos.append(f"{expected}{index + 1}{actual}")
        s7_index = mapping.get(index)
        if s7_index is not None:
            s7pos.append(f"{expected}{s7_index + 1}{actual}")
    return ",".join(seqpos), ",".join(s7pos)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="ProteinMPNN 单运行目录")
    parser.add_argument(
        "--artifacts", default=str(HERE / "artifacts"), help="train_final.py 产物目录"
    )
    parser.add_argument("--chain", default="A", help="设计链名（仅用于标注）")
    parser.add_argument("--candidates", default="", help="候选 CSV；默认自动寻找")
    parser.add_argument("--output", default="", help="输出 CSV；默认 <dir>/scored_candidates.csv")
    args = parser.parse_args(argv)

    run_dir = Path(args.dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"运行目录不存在：{run_dir}")
    candidates_csv = find_candidates(run_dir, args.candidates)
    out_path = Path(args.output) if args.output else run_dir / "scored_candidates.csv"

    bundle = load_bundle(Path(args.artifacts))
    pipeline = bundle["pipeline"]
    valid_codes = bundle["valid_codes"]
    include_aac = bool(bundle["include_aac"])
    include_dpc = bool(bundle["include_dpc"])
    model_name = bundle.get("model_name", "aaindex_nested_cv_auto6")

    reference = wt_construct_356()
    mapping = build_alignment_map()

    table = normalize_candidate_columns(pd.read_csv(candidates_csv))
    if "sequence" not in table.columns:
        raise SystemExit(f"候选表缺少 sequence 列：{candidates_csv}")

    records = []
    for _, row in table.iterrows():
        raw_sequence = str(row.get("sequence", "") or "")
        sequence = "".join(character for character in raw_sequence if character in AA_SET)
        review_status = normalize(row.get("review_status", ""))
        mutation_label_seqpos, mutation_label_s7 = describe_substitutions(
            sequence, reference, mapping
        )

        invalid = 0
        prediction = float("nan")
        if is_blocked(review_status):
            invalid = 2
            print(
                f"[HOLD] 候选 {row.get('source_candidate_id', '?')} 的 review_status="
                f"{review_status}，安全门控拦截（score_invalid=2）",
                file=sys.stderr,
            )
        elif len(sequence) != len(reference):
            invalid = 1
            print(
                f"[WARN] 候选 {row.get('source_candidate_id', '?')} 长度 "
                f"{len(sequence)} != 构建体 {len(reference)}，跳过打分",
                file=sys.stderr,
            )
        else:
            try:
                s7_sequence = construct_to_s7(sequence, mapping)
                features, _ = encode_with_codes(
                    [s7_sequence], valid_codes, include_aac, include_dpc
                )
                reduced = apply_fold(pipeline, features)
                prediction = float(np.asarray(pipeline.model.predict(reduced)).ravel()[0])
            except Exception as exc:  # noqa: BLE001
                invalid = 1
                print(
                    f"[WARN] 候选 {row.get('source_candidate_id', '?')} 打分失败：{exc}",
                    file=sys.stderr,
                )

        records.append(
            {
                "Mutant": mutation_label_s7,
                "Mutant_seqpos": mutation_label_seqpos,
                "Sequence": raw_sequence,
                "Fitness_prediction": prediction,
                "mpnn_score": row.get("mpnn_score", ""),
                "source_candidate_id": row.get("source_candidate_id", ""),
                "review_status": row.get("review_status", ""),
                "mutations_pdb": str(row.get("mutations", "") or ""),
                "score_invalid": invalid,
                "seq_len": len(sequence),
                "model_name": model_name,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(out_path, index=False, float_format="%.6f")
    n_ok = int((out["score_invalid"] == 0).sum())
    n_len = int((out["score_invalid"] == 1).sum())
    n_gate = int((out["score_invalid"] == 2).sum())
    print(f"[OK] 已用 {model_name} 打分 {len(out)} 条 -> {out_path}")
    print(f"     有效(0): {n_ok}  | 长度/映射不符(1): {n_len}  | 审核门控(2): {n_gate}")
    if n_ok:
        valid = out[out["score_invalid"] == 0]["Fitness_prediction"]
        print(
            f"     预测活性范围：{valid.min():.4f} ~ {valid.max():.4f}"
            f"（均值 {valid.mean():.4f}）"
        )
    return 0 if n_ok else 1


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
