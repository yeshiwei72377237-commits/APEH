#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键闭环：aaindex 模型打分（score.py）→ 多样化 TopN 选择（select_top_mutants.py）。

与 src/code/fitness_model/run_loop.py 的调用方式保持一致，方便 run_all.py 直接接入：

    python run_loop.py --dir <run_dir> --artifacts artifacts --chain A --top-k 20

产物：
    <dir>/scored_candidates.csv         每条候选的活性预测
    <dir>/topN_recommendations.csv      去重、多样化后的 TopN 推荐
    <dir>/aaindex_model_run.json        本次打分所用模型与参数的留档（可追溯）
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="ProteinMPNN 单运行目录")
    parser.add_argument(
        "--artifacts", default=str(HERE / "artifacts"), help="train_final.py 产物目录"
    )
    parser.add_argument("--chain", default="A")
    parser.add_argument("--candidates", default="", help="候选 CSV；默认自动寻找")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-pred", type=float, default=0.0)
    parser.add_argument("--focus-lengths", default="4,5,6,7")
    parser.add_argument("--priority-positions", default="")
    parser.add_argument("--per-value-cap", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    args = parser.parse_args(argv)

    run_dir = Path(args.dir).resolve()
    scored = run_dir / "scored_candidates.csv"
    top = run_dir / "topN_recommendations.csv"
    artifacts_dir = Path(args.artifacts).resolve()

    print(f"[1/2] AAindex 模型打分：score.py --dir {run_dir}")
    score_cmd = [
        sys.executable, str(HERE / "score.py"),
        "--dir", str(run_dir),
        "--artifacts", str(artifacts_dir),
        "--chain", args.chain,
    ]
    if args.candidates:
        score_cmd += ["--candidates", args.candidates]
    completed = subprocess.run(score_cmd, check=False)
    if completed.returncode != 0 or not scored.is_file():
        print("[ERR] 打分失败，终止。", file=sys.stderr)
        return 1

    print(f"[2/2] TopN 选择：select_top_mutants.py --top_k {args.top_k}")
    select_cmd = [
        sys.executable, str(HERE / "select_top_mutants.py"),
        "--input_dir", str(run_dir),
        "--pattern", scored.name,
        "--output_csv", str(top),
        "--top_k", str(args.top_k),
        "--min_pred", str(args.min_pred),
        "--focus_lengths", str(args.focus_lengths),
        "--per_value_cap", str(args.per_value_cap),
        "--similarity_threshold", str(args.similarity_threshold),
    ]
    if args.priority_positions:
        select_cmd += ["--priority_positions", args.priority_positions]
    completed = subprocess.run(select_cmd, check=False)
    if completed.returncode != 0 or not top.is_file():
        print("[ERR] TopN 选择失败，终止。", file=sys.stderr)
        return 2

    metadata = {
        "scorer": "aaindex_nested_cv_auto6",
        "scored_csv": str(scored),
        "topn_csv": str(top),
        "top_k": args.top_k,
        "min_pred": args.min_pred,
        "chain": args.chain,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    model_path = artifacts_dir / "final_pipeline.pkl"
    if model_path.is_file():
        with open(model_path, "rb") as handle:
            bundle = pickle.load(handle)
        metadata["model_name"] = bundle.get("model_name", "")
        metadata["algorithm"] = bundle.get("algorithm", "")
        metadata["created"] = bundle.get("created", "")
        metadata["selected_features"] = bundle.get("selected_names", [])
        metadata["audit"] = bundle.get("audit", {})
        metadata["in_sample_metrics"] = bundle.get("in_sample_metrics", {})
    spec_path = artifacts_dir / "feature_spec.json"
    if spec_path.is_file():
        metadata["feature_spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
    (run_dir / "aaindex_model_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] 闭环完成 -> {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
