#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动器：用本包的数据跑 nested_cv.py 的完整嵌套交叉验证（不改动原脚本）。

nested_cv.py 把配置写在文件内部的 USER_CONFIG 里、默认输入 Training_90_Activity.csv，
本启动器只做三件事：指向本包的训练集、指定输出目录与算法/折数，然后调用
其自带的 run_all_algorithms()。输出目录结构与原脚本完全一致。

用法：
    python run_nested_cv.py                                  # 六个算法全跑
    python run_nested_cv.py --algorithms gradientboosting
    python run_nested_cv.py --outer-repeats 2 --out Results_nested_cv

产物：
    <out>/<algorithm>/metric_summary.csv    外层折的无泄漏泛化指标
    <out>/all_algorithms_metric_summary.csv 六算法汇总
    <out>/core_model_comparison.csv         R2 / RMSE / Spearman 对照表
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "nested_cv.py"


def load_script(path: Path):
    if not path.exists():
        raise SystemExit(f"找不到脚本：{path}")
    spec = importlib.util.spec_from_file_location("aaindex_nestedcv", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aaindex_nestedcv"] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default=None, help="逗号分隔，默认六个全跑")
    parser.add_argument("--outer-splits", type=int, default=None)
    parser.add_argument("--outer-repeats", type=int, default=None)
    parser.add_argument("--inner-splits", type=int, default=None)
    parser.add_argument("--out", default="Results_nested_cv", help="输出根目录（相对本包）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_script(SCRIPT)

    config = module.USER_CONFIG
    config["input_csv"] = "data/training_set_activity.csv"
    config["output_root"] = args.out
    if args.outer_splits is not None:
        config["outer_splits"] = args.outer_splits
    if args.outer_repeats is not None:
        config["outer_repeats"] = args.outer_repeats
    if args.inner_splits is not None:
        config["inner_splits"] = args.inner_splits

    available = list(module.ALGORITHMS)
    if args.algorithms:
        chosen = [item.strip() for item in args.algorithms.split(",") if item.strip()]
        unknown = [item for item in chosen if item not in available]
        if unknown:
            raise SystemExit(f"未知算法：{unknown}；可选：{available}")
        module.ALGORITHMS[:] = chosen

    print(f"训练集：{HERE / config['input_csv']}")
    print(f"算法：{module.ALGORITHMS}")
    print(
        f"外层CV：{config['outer_splits']}折 × {config['outer_repeats']}次；"
        f"内层CV：{config['inner_splits']}折"
    )
    print(f"输出根目录：{HERE / config['output_root']}")
    print("-" * 72)

    module.run_all_algorithms()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    main()
