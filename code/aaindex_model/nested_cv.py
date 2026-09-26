#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B：全部 AAindex1 逐位点特征进入无泄漏嵌套交叉验证。

流程
----
CV之前仅做不使用标签的特征计算：
    全部AAindex1 × 序列位置 + 可选AAC/DPC

每个外层训练折内部重新执行：
    中位数填补 → 零方差过滤 → f_regression预筛
    → 相关性去冗余 → SHAP排序 → 内层CV的SFS → 模型拟合

外层测试折只接受训练折已经确定的变换和模型。

说明
----
全部AAindex逐位点编码通常产生十几万维特征。直接在十几万维上计算
相关矩阵、SHAP和SFS不可行，因此先在每个外层训练折内进行监督式预筛。
该预筛严格限制在训练折内，不会造成测试折标签泄漏。

测试命令
--------
python all_aaindex_nested_cv.py \
    --input_csv test_activity.csv \
    --output_dir test_results/all_aaindex_activity \
    --algorithm gradientboosting \
    --outer_splits 5 \
    --outer_repeats 10 \
    --inner_splits 4 \
    --prefilter_k 1000 \
    --corr_top_k 500 \
    --shap_top_k 100 \
    --sfs_min_features 3 \
    --sfs_max_features 15 \
    --include_aac \
    --include_dpc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import shap
except ImportError as exc:
    raise ImportError("请先安装shap：pip install shap") from exc


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
RANDOM_STATE = 2026


# ========================= AAindex兼容读取 =========================

def load_aaindex1():
    try:
        import aaindex  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "未找到aaindex包。请在原来能运行AAindex_2Desc_v2.py的"
            "同一个conda环境中运行。"
        ) from exc

    if hasattr(aaindex, "aaindex1"):
        return aaindex.aaindex1

    try:
        from aaindex import aaindex1  # type: ignore
        return aaindex1
    except Exception as exc:
        raise RuntimeError("当前aaindex包没有可识别的aaindex1接口。") from exc


def list_codes(database) -> List[str]:
    for func in (
        lambda: list(database.record_codes()),
        lambda: list(database.keys()),
    ):
        try:
            values = [str(x) for x in func()]
            if values:
                return sorted(set(values))
        except Exception:
            pass

    if hasattr(database, "parse"):
        database.parse()
        if hasattr(database, "record_codes"):
            values = [str(x) for x in database.record_codes()]
            if values:
                return sorted(set(values))

    raise RuntimeError("无法从当前aaindex包列出AAindex1记录。")


def record_to_mapping(record) -> Dict[str, float]:
    candidates = []
    for attr in ("values", "index", "data"):
        if hasattr(record, attr):
            candidates.append(getattr(record, attr))
    if isinstance(record, dict):
        candidates.append(record)

    for candidate in candidates:
        if isinstance(candidate, dict):
            result = {}
            for aa in AA_ORDER:
                if aa in candidate:
                    try:
                        result[aa] = float(candidate[aa])
                    except Exception:
                        pass
            if len(result) == 20:
                return result

    # 某些包的记录可直接用氨基酸字母索引
    direct = {}
    for aa in AA_ORDER:
        try:
            direct[aa] = float(record[aa])
        except Exception:
            direct = {}
            break
    if len(direct) == 20:
        return direct

    raise RuntimeError("无法把AAindex记录解析成20种氨基酸的数值映射。")


def get_mapping(database, code: str) -> Dict[str, float]:
    record = None
    for func in (
        lambda: database[code],
        lambda: database.get(code),
        lambda: database.record(code),
    ):
        try:
            candidate = func()
            if candidate is not None:
                record = candidate
                break
        except Exception:
            pass

    if record is None:
        raise KeyError(f"无法读取AAindex记录：{code}")

    mapping = record_to_mapping(record)
    if not all(np.isfinite(mapping[aa]) for aa in AA_ORDER):
        raise ValueError(f"{code}含有缺失或非有限值。")
    return mapping


# ========================= 纯序列特征生成 =========================

def clean_sequence(value) -> str:
    sequence = re.sub(r"\s+", "", str(value)).upper()
    if not sequence:
        raise ValueError("发现空序列。")
    invalid = sorted(set(sequence) - AA_SET)
    if invalid:
        raise ValueError("序列含非标准氨基酸：" + ",".join(invalid))
    return sequence


def encode_aac(sequences: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    matrix = np.empty((len(sequences), 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        matrix[i] = [seq.count(aa) / len(seq) for aa in AA_ORDER]
    return matrix, [f"AAC_{aa}" for aa in AA_ORDER]


def encode_dpc(sequences: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    pairs = [a + b for a in AA_ORDER for b in AA_ORDER]
    lookup = {pair: i for i, pair in enumerate(pairs)}
    matrix = np.zeros((len(sequences), 400), dtype=np.float32)

    for row, seq in enumerate(sequences):
        denominator = max(len(seq) - 1, 1)
        for position in range(len(seq) - 1):
            matrix[row, lookup[seq[position:position + 2]]] += 1.0
        matrix[row] /= denominator

    return matrix, [f"DPC_{pair}" for pair in pairs]


def build_all_features(
    sequences: Sequence[str],
    include_aac: bool,
    include_dpc: bool,
    cache_path: Path,
    force_rebuild: bool,
):
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        names = cached["feature_names"].astype(str).tolist()
        valid_codes = cached["valid_codes"].astype(str).tolist()
        skipped = cached["skipped"].astype(str).tolist()
        print(f"读取缓存：{cache_path}")
        print(f"矩阵大小：{X.shape[0]} × {X.shape[1]}")
        return X, names, valid_codes, skipped

    lengths = [len(seq) for seq in sequences]
    if len(set(lengths)) != 1:
        raise ValueError(
            "逐位点AAindex编码要求所有序列等长。长度分布："
            + repr(pd.Series(lengths).value_counts().sort_index().to_dict())
        )

    seq_len = lengths[0]
    aa_to_int = {aa: i for i, aa in enumerate(AA_ORDER)}
    encoded_sequences = np.asarray(
        [[aa_to_int[aa] for aa in seq] for seq in sequences],
        dtype=np.int16,
    )

    database = load_aaindex1()
    codes = list_codes(database)
    print(f"检测到{len(codes)}个AAindex1记录。")

    blocks = []
    feature_names: List[str] = []
    valid_codes: List[str] = []
    skipped: List[str] = []

    started = time.time()
    for number, code in enumerate(codes, start=1):
        try:
            mapping = get_mapping(database, code)
            values = np.asarray(
                [mapping[aa] for aa in AA_ORDER],
                dtype=np.float32,
            )
            block = values[encoded_sequences]
            blocks.append(block)
            feature_names.extend(
                f"{code}_pos_{position}"
                for position in range(1, seq_len + 1)
            )
            valid_codes.append(code)
        except Exception as exc:
            skipped.append(f"{code}: {exc}")

        if number % 50 == 0 or number == len(codes):
            print(
                f"AAindex进度 {number}/{len(codes)}；"
                f"成功{len(valid_codes)}；跳过{len(skipped)}"
            )

    if include_aac:
        block, names = encode_aac(sequences)
        blocks.append(block)
        feature_names.extend(names)

    if include_dpc:
        block, names = encode_dpc(sequences)
        blocks.append(block)
        feature_names.extend(names)

    if not blocks:
        raise RuntimeError("没有成功生成任何特征。")

    X = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        feature_names=np.asarray(feature_names, dtype=object),
        valid_codes=np.asarray(valid_codes, dtype=object),
        skipped=np.asarray(skipped, dtype=object),
    )

    print(
        f"特征生成完成：{X.shape[0]} × {X.shape[1]}；"
        f"耗时{(time.time() - started) / 60:.2f}分钟"
    )
    print(f"缓存保存至：{cache_path}")
    return X, feature_names, valid_codes, skipped


# ========================= 模型与指标 =========================

def build_model(name: str, seed: int) -> BaseEstimator:
    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
    if name == "gradientboosting":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=seed,
        )
    if name == "adaboost":
        return AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.03,
            random_state=seed,
        )
    if name == "bagging":
        return BaggingRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=1,
        )
    if name == "svr":
        return SVR(C=10.0, epsilon=0.1, gamma="scale")
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("使用xgb请先安装xgboost。") from exc
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"不支持的算法：{name}")


def safe_spearman(y_true, y_pred) -> float:
    value = spearmanr(y_true, y_pred).correlation
    return float(value) if np.isfinite(value) else np.nan


def spearman_scorer(estimator, X, y) -> float:
    value = safe_spearman(y, estimator.predict(X))
    return 0.0 if not np.isfinite(value) else value


def get_scoring(name: str):
    return {
        "spearman": spearman_scorer,
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
    }[name]


# ========================= 训练折内筛选 =========================

def select_f_top_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, _ = f_regression(X_train, y_train, center=True)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max)
    k = min(max(1, k), X_train.shape[1])

    if k == X_train.shape[1]:
        indices = np.arange(X_train.shape[1])
    else:
        indices = np.argpartition(scores, -k)[-k:]

    indices = indices[np.argsort(scores[indices])[::-1]]
    return indices, scores[indices]


def correlation_keep_indices(
    X_train: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if X_train.shape[1] <= 1:
        return np.arange(X_train.shape[1])

    corr = np.abs(np.corrcoef(X_train, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)

    keep = []
    for index in range(X_train.shape[1]):
        if not keep or np.all(corr[index, keep] <= threshold):
            keep.append(index)
    return np.asarray(keep, dtype=int)


def rank_by_shap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    names: Sequence[str],
    estimator: BaseEstimator,
    max_background: int,
    max_explain: int,
    seed: int,
) -> pd.DataFrame:
    model = clone(estimator)
    model.fit(X_train, y_train)

    rng = np.random.default_rng(seed)
    bg_idx = (
        np.arange(len(X_train))
        if len(X_train) <= max_background
        else rng.choice(len(X_train), max_background, replace=False)
    )
    ex_idx = (
        np.arange(len(X_train))
        if len(X_train) <= max_explain
        else rng.choice(len(X_train), max_explain, replace=False)
    )

    background = X_train[bg_idx]
    explain_data = X_train[ex_idx]

    try:
        explainer = shap.Explainer(model, background)
        explanation = explainer(explain_data)
    except Exception:
        explainer = shap.Explainer(model.predict, background)
        explanation = explainer(explain_data)

    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise RuntimeError(f"SHAP输出维度异常：{values.shape}")

    result = pd.DataFrame({
        "Feature": list(names),
        "MeanAbsSHAP": np.mean(np.abs(values), axis=0),
        "MeanSignedSHAP": np.mean(values, axis=0),
    }).sort_values("MeanAbsSHAP", ascending=False, kind="mergesort")
    result["SHAP_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def run_sfs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    names: Sequence[str],
    estimator: BaseEstimator,
    inner_cv: KFold,
    min_features: int,
    max_features: int,
    scoring_name: str,
    tolerance: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    scoring = get_scoring(scoring_name)
    n = X_train.shape[1]
    min_features = min(max(1, min_features), n)
    max_features = min(max(min_features, max_features), n)

    selected: List[int] = []
    remaining = list(range(n))
    history = []

    for step in range(1, max_features + 1):
        candidates = []
        for candidate in remaining:
            subset = selected + [candidate]
            scores = cross_val_score(
                clone(estimator),
                X_train[:, subset],
                y_train,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
            candidates.append((
                candidate,
                float(np.mean(scores)),
                float(np.std(scores, ddof=1)),
            ))

        candidates.sort(key=lambda item: (-item[1], names[item[0]]))
        best, mean_score, sd_score = candidates[0]
        selected.append(best)
        remaining.remove(best)

        history.append({
            "Step": step,
            "Added_feature": names[best],
            "Selected_features": json.dumps(
                [names[i] for i in selected],
                ensure_ascii=False,
            ),
            "N_features": len(selected),
            "Mean_inner_score": mean_score,
            "SD_inner_score": sd_score,
        })

    history_df = pd.DataFrame(history)
    eligible = history_df[
        history_df["N_features"].between(min_features, max_features)
    ]
    best_score = eligible["Mean_inner_score"].max()
    chosen = eligible[
        eligible["Mean_inner_score"] >= best_score - tolerance
    ].sort_values(
        ["N_features", "Mean_inner_score"],
        ascending=[True, False],
    ).iloc[0]

    chosen_size = int(chosen["N_features"])
    history_df["Chosen_subset"] = (
        history_df["N_features"] == chosen_size
    )
    return np.asarray(selected[:chosen_size], dtype=int), history_df


@dataclass
class FoldPipeline:
    imputer: SimpleImputer
    variance_mask: np.ndarray
    prefilter_indices: np.ndarray
    scaler: StandardScaler
    selected_prefilter_indices: np.ndarray
    selected_names: List[str]
    model: BaseEstimator


def fit_fold(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    all_names: Sequence[str],
    args,
    seed: int,
):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)

    variance = np.var(X_train, axis=0)
    variance_mask = np.isfinite(variance) & (variance > 0)
    X_variance = X_train[:, variance_mask]
    variance_names = np.asarray(all_names, dtype=object)[variance_mask]

    prefilter_indices, f_scores = select_f_top_k(
        X_variance, y_train, args.prefilter_k
    )
    X_prefilter = X_variance[:, prefilter_indices]
    prefilter_names = variance_names[prefilter_indices].astype(str)

    scaler = StandardScaler()
    X_prefilter = scaler.fit_transform(X_prefilter)

    corr_input_n = min(args.corr_top_k, X_prefilter.shape[1])
    corr_input = X_prefilter[:, :corr_input_n]
    corr_names = prefilter_names[:corr_input_n]
    corr_keep = correlation_keep_indices(
        corr_input, args.corr_threshold
    )
    X_corr = corr_input[:, corr_keep]
    corr_names = corr_names[corr_keep].astype(str).tolist()

    estimator = build_model(args.algorithm, seed)
    shap_ranking = rank_by_shap(
        X_corr,
        y_train,
        corr_names,
        estimator,
        args.shap_max_background,
        args.shap_max_explain,
        seed,
    )

    shap_n = min(args.shap_top_k, len(shap_ranking))
    shap_names = shap_ranking.head(shap_n)["Feature"].tolist()
    corr_name_to_index = {name: i for i, name in enumerate(corr_names)}
    shap_corr_indices = np.asarray(
        [corr_name_to_index[name] for name in shap_names],
        dtype=int,
    )
    X_shap = X_corr[:, shap_corr_indices]

    inner_cv = KFold(
        n_splits=min(args.inner_splits, len(y_train)),
        shuffle=True,
        random_state=seed + 1000,
    )
    sfs_local, sfs_history = run_sfs(
        X_shap,
        y_train,
        shap_names,
        estimator,
        inner_cv,
        args.sfs_min_features,
        args.sfs_max_features,
        args.sfs_scoring,
        args.sfs_tolerance,
    )

    # 最终特征在“预筛后标准化矩阵”中的列位置
    selected_prefilter_indices = corr_keep[
        shap_corr_indices[sfs_local]
    ]
    selected_names = [shap_names[i] for i in sfs_local]

    final_model = clone(estimator)
    final_model.fit(
        X_prefilter[:, selected_prefilter_indices],
        y_train,
    )

    pipeline = FoldPipeline(
        imputer=imputer,
        variance_mask=variance_mask,
        prefilter_indices=prefilter_indices,
        scaler=scaler,
        selected_prefilter_indices=selected_prefilter_indices,
        selected_names=selected_names,
        model=final_model,
    )

    audit = {
        "N_original": X_train_raw.shape[1],
        "N_after_variance": int(np.sum(variance_mask)),
        "N_prefilter": len(prefilter_indices),
        "N_corr_input": corr_input_n,
        "N_after_correlation": len(corr_keep),
        "N_SHAP_candidates": shap_n,
        "N_final_features": len(selected_names),
    }
    return pipeline, shap_ranking, sfs_history, audit


def apply_fold(pipeline: FoldPipeline, X_raw: np.ndarray) -> np.ndarray:
    X = pipeline.imputer.transform(X_raw)
    X = X[:, pipeline.variance_mask]
    X = X[:, pipeline.prefilter_indices]
    X = pipeline.scaler.transform(X)
    return X[:, pipeline.selected_prefilter_indices]


# ========================= 评价与输出 =========================

def metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mse)),
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
    }


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["R2", "RMSE", "MSE", "MAE", "Spearman"]:
        values = pd.to_numeric(table[metric], errors="coerce").dropna()
        n = len(values)
        mean = values.mean()
        sd = values.std(ddof=1) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        rows.append({
            "Metric": metric,
            "N_outer_folds": n,
            "Mean": mean,
            "SD": sd,
            "CI95_low": mean - 1.96 * sem if n > 1 else np.nan,
            "CI95_high": mean + 1.96 * sem if n > 1 else np.nan,
            "Median": values.median(),
        })
    return pd.DataFrame(rows)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input_csv)

    for column in (
        args.id_column,
        args.sequence_column,
        args.target_column,
    ):
        if column not in data.columns:
            raise ValueError(f"输入文件缺少列：{column}")

    sample_ids = data[args.id_column].astype(str).to_numpy()
    sequences = [clean_sequence(x) for x in data[args.sequence_column]]
    y = pd.to_numeric(
        data[args.target_column], errors="raise"
    ).to_numpy(float)

    cache = (
        args.feature_cache
        if args.feature_cache
        else args.output_dir / "all_aaindex_features.npz"
    )
    X, names, valid_codes, skipped = build_all_features(
        sequences,
        args.include_aac,
        args.include_dpc,
        cache,
        args.force_rebuild_features,
    )

    if X.shape[0] != len(y):
        raise RuntimeError("特征矩阵与标签样本数不一致。")

    outer_cv = RepeatedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )

    fold_rows = []
    prediction_rows = []
    selection_rows = []
    shap_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        repeat = (fold - 1) // args.outer_splits + 1
        split = (fold - 1) % args.outer_splits + 1
        fold_dir = args.output_dir / f"fold_{fold:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pipeline, ranking, sfs_history, audit = fit_fold(
            X[train_idx],
            y[train_idx],
            names,
            args,
            args.random_state + fold,
        )
        X_test_final = apply_fold(pipeline, X[test_idx])
        y_pred = pipeline.model.predict(X_test_final)
        fold_metrics = metrics(y[test_idx], y_pred)

        fold_rows.append({
            "Fold": fold,
            "Repeat": repeat,
            "Split": split,
            "Algorithm": args.algorithm,
            "N_train": len(train_idx),
            "N_test": len(test_idx),
            **audit,
            **fold_metrics,
        })

        rank_lookup = dict(zip(
            ranking["Feature"], ranking["SHAP_rank"]
        ))
        shap_lookup = dict(zip(
            ranking["Feature"], ranking["MeanAbsSHAP"]
        ))
        for order, feature in enumerate(
            pipeline.selected_names, start=1
        ):
            selection_rows.append({
                "Fold": fold,
                "Repeat": repeat,
                "Split": split,
                "Feature": feature,
                "Selection_order": order,
                "SHAP_rank": rank_lookup.get(feature, np.nan),
                "MeanAbsSHAP": shap_lookup.get(feature, np.nan),
            })

        fold_ranking = ranking.copy()
        fold_ranking.insert(0, "Split", split)
        fold_ranking.insert(0, "Repeat", repeat)
        fold_ranking.insert(0, "Fold", fold)
        shap_rows.extend(fold_ranking.to_dict("records"))

        for sid, observed, predicted in zip(
            sample_ids[test_idx], y[test_idx], y_pred
        ):
            prediction_rows.append({
                "Fold": fold,
                "Repeat": repeat,
                "Split": split,
                "Sample_ID": sid,
                "Observed": observed,
                "Predicted": predicted,
                "Residual": observed - predicted,
            })

        ranking.to_csv(
            fold_dir / "shap_ranking.csv", index=False
        )
        sfs_history.to_csv(
            fold_dir / "sfs_search_history.csv", index=False
        )
        pd.DataFrame({
            "Feature": pipeline.selected_names,
            "Selection_order": range(
                1, len(pipeline.selected_names) + 1
            ),
        }).to_csv(
            fold_dir / "selected_features.csv", index=False
        )

        print(
            f"Fold {fold:03d}: "
            f"R2={fold_metrics['R2']:.4f}, "
            f"RMSE={fold_metrics['RMSE']:.4f}, "
            f"Spearman={fold_metrics['Spearman']:.4f}, "
            f"features={audit['N_final_features']}"
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    all_shap = pd.DataFrame(shap_rows)

    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(
        args.output_dir / "out_of_fold_predictions.csv", index=False
    )
    selections.to_csv(
        args.output_dir / "selected_features_by_fold.csv", index=False
    )
    all_shap.to_csv(
        args.output_dir / "shap_rankings_by_fold.csv", index=False
    )
    summary = summarize(folds)
    summary.to_csv(
        args.output_dir / "metric_summary.csv", index=False
    )

    if not selections.empty:
        frequency = (
            selections.groupby("Feature", as_index=False)
            .agg(
                Selection_count=("Fold", "nunique"),
                Mean_SHAP_rank=("SHAP_rank", "mean"),
                Median_SHAP_rank=("SHAP_rank", "median"),
            )
            .sort_values(
                ["Selection_count", "Mean_SHAP_rank"],
                ascending=[False, True],
            )
        )
        frequency["Selection_frequency"] = (
            frequency["Selection_count"] / len(folds)
        )
    else:
        frequency = pd.DataFrame()
    frequency.to_csv(
        args.output_dir / "feature_selection_frequency.csv",
        index=False,
    )

    metadata = {
        "input_csv": str(args.input_csv),
        "n_samples": len(y),
        "sequence_length": len(sequences[0]),
        "n_valid_aaindex_records": len(valid_codes),
        "n_skipped_aaindex_records": len(skipped),
        "n_total_features": X.shape[1],
        "feature_cache": str(cache),
        "algorithm": args.algorithm,
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits,
        "prefilter_k": args.prefilter_k,
        "corr_top_k": args.corr_top_k,
        "shap_top_k": args.shap_top_k,
        "sfs_min_features": args.sfs_min_features,
        "sfs_max_features": args.sfs_max_features,
        "statement": (
            "All supervised prefiltering, correlation pruning, SHAP "
            "ranking and SFS were fitted independently in each outer "
            "training fold."
        ),
        "skipped_aaindex_records": skipped,
    }
    with open(
        args.output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("\n运行完成：")
    print(summary.to_string(index=False))
    print(f"\n结果目录：{args.output_dir.resolve()}")



# ========================= 一键运行 6 个算法 =========================
# 使用方法：
# 1. 把本脚本与 Training_90_Activity.csv 放在同一文件夹；
# 2. 在 VS Code / PyCharm 中直接点击“运行”即可；
# 3. 无需在终端输入任何参数。
#
# 如需调整训练设置，只修改下面 USER_CONFIG 即可。

USER_CONFIG = {
    # 输入文件：默认读取脚本所在目录下的 Training_90_Activity.csv
    "input_csv": "Training_90_Activity.csv",

    # 总结果文件夹：六个算法会分别建立子目录
    "output_root": "Results_90_Activity_All",

    # 输入 CSV 中的列名
    "target_column": "Fitness",
    "id_column": "Sequence ID",
    "sequence_column": "Sequence",

    # 嵌套交叉验证参数
    "outer_splits": 5,
    "outer_repeats": 10,
    "inner_splits": 4,

    # 特征筛选参数
    "prefilter_k": 1000,
    "corr_top_k": 500,
    "corr_threshold": 0.95,
    "shap_top_k": 100,
    "shap_max_background": 80,
    "shap_max_explain": 120,
    "sfs_min_features": 5,
    "sfs_max_features": 15,
    "sfs_scoring": "spearman",
    "sfs_tolerance": 0.0,

    # 是否加入 AAC / DPC 特征
    "include_aac": True,
    "include_dpc": True,

    # 随机种子
    "random_state": RANDOM_STATE,
}


ALGORITHMS = [
    "randomforest",
    "gradientboosting",
    "adaboost",
    "bagging",
    "svr",
    "xgb",
]


def make_run_args(
    algorithm: str,
    script_dir: Path,
    output_root: Path,
    shared_cache: Path,
    force_rebuild_features: bool,
):
    """把 USER_CONFIG 转换成原 main() 可直接使用的参数对象。"""
    cfg = USER_CONFIG
    return argparse.Namespace(
        input_csv=script_dir / cfg["input_csv"],
        output_dir=output_root / algorithm,
        target_column=cfg["target_column"],
        id_column=cfg["id_column"],
        sequence_column=cfg["sequence_column"],
        feature_cache=shared_cache,
        force_rebuild_features=force_rebuild_features,
        algorithm=algorithm,
        outer_splits=cfg["outer_splits"],
        outer_repeats=cfg["outer_repeats"],
        inner_splits=cfg["inner_splits"],
        prefilter_k=cfg["prefilter_k"],
        corr_top_k=cfg["corr_top_k"],
        corr_threshold=cfg["corr_threshold"],
        shap_top_k=cfg["shap_top_k"],
        shap_max_background=cfg["shap_max_background"],
        shap_max_explain=cfg["shap_max_explain"],
        sfs_min_features=cfg["sfs_min_features"],
        sfs_max_features=cfg["sfs_max_features"],
        sfs_scoring=cfg["sfs_scoring"],
        sfs_tolerance=cfg["sfs_tolerance"],
        include_aac=cfg["include_aac"],
        include_dpc=cfg["include_dpc"],
        random_state=cfg["random_state"],
    )


def collect_algorithm_comparison(output_root: Path) -> pd.DataFrame:
    """汇总六个算法各自的 metric_summary.csv，方便直接比较。"""
    rows = []
    for algorithm in ALGORITHMS:
        summary_file = output_root / algorithm / "metric_summary.csv"
        if not summary_file.exists():
            continue
        table = pd.read_csv(summary_file)
        for _, row in table.iterrows():
            rows.append({
                "Algorithm": algorithm,
                "Metric": row.get("Metric"),
                "N_outer_folds": row.get("N_outer_folds"),
                "Mean": row.get("Mean"),
                "SD": row.get("SD"),
                "CI95_low": row.get("CI95_low"),
                "CI95_high": row.get("CI95_high"),
                "Median": row.get("Median"),
            })

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison.to_csv(
            output_root / "all_algorithms_metric_summary.csv",
            index=False,
        )
    return comparison


def run_all_algorithms():
    """无需命令行参数，一次依次运行六种算法。"""
    script_dir = Path(__file__).resolve().parent
    input_csv = script_dir / USER_CONFIG["input_csv"]
    output_root = script_dir / USER_CONFIG["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{input_csv}\n"
            "请把 Training_90_Activity.csv 与本脚本放在同一文件夹，"
            "或修改 USER_CONFIG['input_csv']。"
        )

    if USER_CONFIG["outer_splits"] < 2:
        raise ValueError("outer_splits 必须至少为 2。")
    if USER_CONFIG["shap_top_k"] < USER_CONFIG["sfs_max_features"]:
        raise ValueError("shap_top_k 不能小于 sfs_max_features。")
    if USER_CONFIG["prefilter_k"] < USER_CONFIG["corr_top_k"]:
        warnings.warn(
            "prefilter_k 小于 corr_top_k，实际相关性输入会受 prefilter_k 限制。"
        )

    # 六个算法使用完全相同的原始序列特征，因此共用一个缓存文件。
    # 第一个算法强制重建一次，之后五个算法直接读取该缓存。
    shared_cache = output_root / "shared_all_aaindex_features.npz"

    print("=" * 72)
    print("开始自动运行 6 个回归算法")
    print(f"输入文件：{input_csv}")
    print(f"结果目录：{output_root}")
    print(
        f"外层CV：{USER_CONFIG['outer_splits']}折 × "
        f"{USER_CONFIG['outer_repeats']}次重复；"
        f"内层CV：{USER_CONFIG['inner_splits']}折"
    )
    print("算法顺序：" + " -> ".join(ALGORITHMS))
    print("=" * 72)

    failures = []

    for index, algorithm in enumerate(ALGORITHMS, start=1):
        print("\n" + "#" * 72)
        print(f"[{index}/{len(ALGORITHMS)}] 正在运行：{algorithm}")
        print("#" * 72)

        args = make_run_args(
            algorithm=algorithm,
            script_dir=script_dir,
            output_root=output_root,
            shared_cache=shared_cache,
            force_rebuild_features=(index == 1),
        )

        try:
            main(args)
        except Exception as exc:
            failures.append({
                "Algorithm": algorithm,
                "Error_type": type(exc).__name__,
                "Error": str(exc),
            })
            print(f"\n!!! {algorithm} 运行失败：{type(exc).__name__}: {exc}")
            print("脚本将继续尝试下一个算法。")

    comparison = collect_algorithm_comparison(output_root)

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_df.to_csv(output_root / "failed_algorithms.csv", index=False)
        print("\n有算法运行失败，详情已保存：")
        print(output_root / "failed_algorithms.csv")

    print("\n" + "=" * 72)
    print("六算法批量运行结束。")
    if not comparison.empty:
        print("综合指标汇总：")
        print(output_root / "all_algorithms_metric_summary.csv")

        # 额外输出一个更易阅读的核心指标表
        core = comparison[comparison["Metric"].isin(["R2", "RMSE", "Spearman"])].copy()
        if not core.empty:
            pivot = core.pivot(index="Algorithm", columns="Metric", values="Mean")
            preferred = [c for c in ["R2", "RMSE", "Spearman"] if c in pivot.columns]
            pivot = pivot[preferred]
            pivot.to_csv(output_root / "core_model_comparison.csv")
            print("核心模型比较：")
            print(output_root / "core_model_comparison.csv")
            print("\n平均性能：")
            print(pivot.to_string())

    print(f"\n总结果目录：{output_root.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    run_all_algorithms()
