#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键闭环：对 MPNN 候选先打分(score.py)，再按多样化/长度约束输出 TopN(select_top_mutants.py)。

用法示例:
    python src/code/fitness_model/run_loop.py \\
        --dir <run_dir> \\
        --artifacts src/code/fitness_model/artifacts \\
        --chain A --top-k 20 --min-pred <阈值>

产物：
    <dir>/scored_candidates.csv         每条候选的 Fitness_prediction + 序列位置 Mutant
    <dir>/topN_recommendations.csv      最终 TopN 推荐（去重、多样化、长度聚焦）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="ProteinMPNN 单运行目录")
    ap.add_argument("--artifacts",
                    default=str(HERE / "artifacts"),
                    help="train.py 产出的 artifacts 目录")
    ap.add_argument("--chain", default="A")
    ap.add_argument("--candidates", default="", help="候选 CSV；默认自动寻找")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-pred", type=float, default=0.0)
    ap.add_argument("--focus-lengths", default="4,5,6,7")
    ap.add_argument("--priority-positions", default="")
    ap.add_argument("--per-value-cap", type=int, default=5)
    ap.add_argument("--similarity-threshold", type=float, default=0.8)
    ap.add_argument("--best-mpnn", action="store_true",
                    help="若候选里含 mpnn_score，可为以后扩展；当前仅作参考")
    args = ap.parse_args(argv)

    run_dir = Path(args.dir).resolve()
    scored = run_dir / "scored_candidates.csv"
    top = run_dir / "topN_recommendations.csv"

    print(f"[1/2] 打分: score.py --dir {run_dir}")
    score_cmd = [
        sys.executable, str(HERE / "score.py"),
        "--dir", str(run_dir),
        "--artifacts", str(Path(args.artifacts).resolve()),
        "--chain", args.chain,
    ]
    if args.candidates:
        score_cmd += ["--candidates", args.candidates]
    r1 = subprocess.run(score_cmd, check=False)
    if r1.returncode != 0 or not scored.is_file():
        print("[ERR] 打分失败，终止。", file=sys.stderr)
        return 1

    print(f"[2/2] TopN 选择: select_top_mutants.py --input_dir {run_dir} --top_k {args.top_k}")
    sel_cmd = [
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
        sel_cmd += ["--priority_positions", args.priority_positions]
    r2 = subprocess.run(sel_cmd, check=False)
    if r2.returncode != 0 or not top.is_file():
        print("[ERR] TopN 选择失败，终止。", file=sys.stderr)
        return 2

    print(f"[OK] 闭环完成 → {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
