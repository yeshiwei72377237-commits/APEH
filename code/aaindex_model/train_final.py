#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在全部训练样本上冻结一份 nested_cv.py 的流程，产出可部署的打分模型。

与 nested_cv.py 的关系
----------------------
特征构建、折内筛选（中位数填补 → 零方差过滤 → f_regression 预筛 →
相关性去冗余 → SHAP 排序 → 内层 CV 的 SFS）与估计器构造，全部**直接调用
nested_cv.py 的同名函数**，因此部署模型与嵌套 CV 中评估的模型是同一套流程。
唯一区别：这里在 100% 训练样本上拟合一次（部署需要单一模型），而
nested_cv.py 只在各外层训练折内拟合、用于给出无泄漏的泛化估计。

用法
----
    python train_final.py                       # 用 config.json 的默认算法
    python train_final.py --algorithm xgb       # 换算法重新冻结
    python train_final.py --cv-summary <metric_summary.csv>   # 附带嵌套CV指标

产物（artifacts/）
-----------------
    final_pipeline.pkl     冻结的 FoldPipeline + 特征规格 + 训练元信息
    feature_spec.json      人类可读的特征/超参摘要
    training_metrics.json  训练集内表现（附嵌套 CV 结果如有）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nested_cv import (  # noqa: E402
    apply_fold,
    build_all_features,
    fit_fold,
    metrics,
)

ARTIFACTS = HERE / "artifacts"
DATA = HERE / "data"


def configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_config(path: str | None) -> dict:
    config_path = Path(path) if path else HERE / "config.json"
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else HERE / path


def build_run_args(config: dict, algorithm: str) -> argparse.Namespace:
    """把 config.json 转成 nested_cv.fit_fold 直接可用的参数对象。"""
    return argparse.Namespace(
        algorithm=algorithm,
        prefilter_k=config["prefilter_k"],
        corr_top_k=config["corr_top_k"],
        corr_threshold=config["corr_threshold"],
        inner_splits=config["inner_splits"],
        shap_top_k=config["shap_top_k"],
        shap_max_background=config["shap_max_background"],
        shap_max_explain=config["shap_max_explain"],
        sfs_min_features=config["sfs_min_features"],
        sfs_max_features=config["sfs_max_features"],
        sfs_scoring=config["sfs_scoring"],
        sfs_tolerance=config["sfs_tolerance"],
    )


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="模型配置 json（默认包内 config.json）")
    parser.add_argument("--algorithm", default="", help="覆盖 config.json 的 algorithm")
    parser.add_argument("--output", default=str(ARTIFACTS), help="产物目录")
    parser.add_argument(
        "--cache",
        default="",
        help="特征缓存 npz 路径；默认放在系统临时目录（避免仓库里出现上百 MB 文件）",
    )
    parser.add_argument(
        "--keep-cache", action="store_true", help="保留特征缓存（重复训练时加速）"
    )
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument(
        "--cv-summary",
        default="",
        help="可选：nested_cv.py 产出的 metric_summary.csv，用于在产物中记录无泄漏泛化指标",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config or None)
    algorithm = args.algorithm or config["algorithm"]
    seed = int(config["random_state"])

    training_csv = resolve(config["training_csv"])
    if not training_csv.is_file():
        raise SystemExit(f"找不到训练集：{training_csv}")

    data = pd.read_csv(training_csv)
    for column in (
        config["id_column"],
        config["sequence_column"],
        config["target_column"],
    ):
        if column not in data.columns:
            raise SystemExit(f"训练集缺少列：{column}")

    sequences = [str(value) for value in data[config["sequence_column"]]]
    y = pd.to_numeric(data[config["target_column"]], errors="raise").to_numpy(float)

    cache = Path(args.cache) if args.cache else Path(tempfile.gettempdir()) / (
        "aaindex_train_features_"
        + hashlib.sha1(str(training_csv).encode("utf-8")).hexdigest()[:8]
        + ".npz"
    )

    print(f"训练集：{training_csv}")
    print(f"样本数：{len(y)}；算法：{algorithm}；随机种子：{seed}")
    X, names, valid_codes, skipped = build_all_features(
        sequences,
        bool(config["include_aac"]),
        bool(config["include_dpc"]),
        cache,
        args.force_rebuild_features,
    )
    if X.shape[0] != len(y):
        raise SystemExit("特征矩阵与标签样本数不一致。")
    print(f"特征矩阵：{X.shape[0]} × {X.shape[1]}（可用 AAindex 记录 {len(valid_codes)} 条）")

    run_args = build_run_args(config, algorithm)
    print("正在全部训练样本上拟合冻结流程（填补→零方差→预筛→去冗余→SHAP→SFS）……")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline, ranking, sfs_history, audit = fit_fold(
            X, y, names, run_args, seed
        )

    X_final = apply_fold(pipeline, X)
    y_pred = pipeline.model.predict(X_final)
    in_sample = metrics(y, y_pred)
    residuals = y - y_pred

    print(f"训练集内表现：R²={in_sample['R2']:.4f}  RMSE={in_sample['RMSE']:.4f}  "
          f"Spearman={in_sample['Spearman']:.4f}  特征数={audit['N_final_features']}")
    print("（训练集内指标偏乐观；无泄漏泛化估计请看 nested_cv.py 的 metric_summary.csv）")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": config["model_name"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "pipeline": pipeline,
        "algorithm": algorithm,
        "random_state": seed,
        "valid_codes": list(valid_codes),
        "skipped_codes": list(skipped),
        "include_aac": bool(config["include_aac"]),
        "include_dpc": bool(config["include_dpc"]),
        "n_features_total": int(X.shape[1]),
        "sequence_length": int(X.shape[1] and len(sequences[0])),
        "train_sequences": int(len(y)),
        "audit": audit,
        "selected_names": list(pipeline.selected_names),
        "hyperparams": {
            key: config[key]
            for key in (
                "prefilter_k",
                "corr_top_k",
                "corr_threshold",
                "shap_top_k",
                "shap_max_background",
                "shap_max_explain",
                "sfs_min_features",
                "sfs_max_features",
                "sfs_scoring",
                "sfs_tolerance",
            )
        },
        "in_sample_metrics": in_sample,
        "training_csv": str(training_csv),
        "training_csv_sha256": sha256_of(training_csv),
        "source_script_sha256": sha256_of(HERE / "nested_cv.py"),
        "residual_summary": {
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals, ddof=1)),
        },
    }
    with open(output_dir / "final_pipeline.pkl", "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    feature_spec = {
        "model_name": config["model_name"],
        "source_script": config["source_script"],
        "source_script_sha256": bundle["source_script_sha256"],
        "task": "xylanase activity regression",
        "sequence_space": config["sequence_space"],
        "encoding": "nested_cv.build_all_features（全部 AAindex1 逐位点 + AAC + DPC）",
        "algorithm": algorithm,
        "random_state": seed,
        "n_sequences_train": int(len(y)),
        "n_features_total": int(X.shape[1]),
        "n_aaindex_codes_used": len(valid_codes),
        "n_aaindex_codes_skipped": len(skipped),
        "audit": audit,
        "selected_features": list(pipeline.selected_names),
        "hyperparams": bundle["hyperparams"],
        "provenance": {
            "source_script_path": "D:/APEH2.0/Xylanase_2/all_aaindex_nested_cv_auto6(1).py",
            "note": "本包内的 nested_cv.py 与该脚本逐字节一致（SHA256 见上）。",
        },
    }
    with open(output_dir / "feature_spec.json", "w", encoding="utf-8") as handle:
        json.dump(feature_spec, handle, ensure_ascii=False, indent=2)

    training_metrics = {
        "model_name": config["model_name"],
        "created": bundle["created"],
        "algorithm": algorithm,
        "in_sample": in_sample,
        "residual_summary": bundle["residual_summary"],
        "note": (
            "in_sample 是在全部训练样本上拟合后再预测同一批样本，偏乐观；"
            "请以 nested_cv.py 的外层折指标作为泛化估计。"
        ),
    }
    if args.cv_summary:
        summary_path = Path(args.cv_summary)
        if summary_path.is_file():
            table = pd.read_csv(summary_path)
            training_metrics["nested_cv"] = table.to_dict("records")
            training_metrics["nested_cv_source"] = str(summary_path)
        else:
            print(f"[WARN] 找不到 --cv-summary 指定的文件：{summary_path}")
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(training_metrics, handle, ensure_ascii=False, indent=2)

    print(f"产物目录：{output_dir}")
    print("  final_pipeline.pkl / feature_spec.json / training_metrics.json")

    if not args.keep_cache and cache.is_file():
        try:
            cache.unlink()
            print(f"已删除特征缓存（--keep-cache 可保留）：{cache}")
        except OSError as exc:
            print(f"[WARN] 删除特征缓存失败：{exc}")

    if not math.isfinite(in_sample["Spearman"]):
        print("[WARN] Spearman 非有限值，请检查数据。")
    return 0


if __name__ == "__main__":
    configure_streams()
    raise SystemExit(main())
