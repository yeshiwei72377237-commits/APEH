#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在湿实验活性数据上训练 fitness 回归模型，并把训练产物落盘到 artifacts/。

用法示例:
    python src/code/fitness_model/train.py \\
        --xlsx data/raw/Supplementary-Material-2.xlsx \\
        --sheet Sheet1 --target activity

训练产物（--out，默认 artifacts/）：
    model.pkl            sklearn GradientBoostingRegressor (保存 model + featurizer 配置)
    featurizer_meta.json 重建 SequenceFeaturizer 所需的 features 配置
    wt.txt               训练所用 WT 序列（供 score.py 做蛋白特异校验）
    target_meta.json     目标列名/样本数/模型参数等元信息

说明：这是「快速闭环基线」。默认训练集为 Supplementary Material-2（126 条标签、
测量噪声大，R² 属入门级），不代表高精度；训练集/目标列可用 --xlsx/--target 切换。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 修正 Windows 控制台的 GBK/utf-8 编码冲突（见 run_all.py 同款处理）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from featurize import SequenceFeaturizer, AA_LIST

# 训练数据对应的 360-aa 野生型木聚糖酶序列（默认值；可用 --wt 或 --wt-file 覆盖）
DEFAULT_WT = (
    "GVFGENQKRNDQPFAWQVASLSERYQEQFDIGAAVEPYQLEGRQAQILKHHYNSLVAENAMKPVSLQPRE"
    "GEWNWEGADKIVEFARKHNMELRFHTLVWHSQVPEWFFIDENGNRMVDETDPEKRKANKQLLLERMENHI"
    "KTVVERYKDDVTSWDVVNEVIDDGGGLRESEWYQITGTDYIKVAFETARKYGGEEAKLYINDYNTEVPSK"
    "RDDLYNLVKDLLEQGVPIDGVGHQSHIQIGWPSIEDTRASFEKFTSLGLDNQVTELDMSLYGWPPTGAYTS"
    "YDDIPEELFQAQADRYDQLFELYEELSATISSVTFWGIADNHTWLDDRAREYNNGVGVDAPFVFDHNYRVK"
    "PAYWRIID"
)

_TOKEN_RE = re.compile(r"[A-Za-z]\d+[A-Za-z]")
_AA_SET = set(AA_LIST)


def parse_tokens(site: str) -> list[str]:
    """从 'A127W-V160A-V207N-Q238S' 提取 token 列表；只保留标准 AA 格式。"""
    if not site:
        return []
    return [t for t in _TOKEN_RE.findall(str(site)) if t]


def mut2seq(wt: str, tokens: list[str]) -> str:
    """把 token 应用到 WT，返回 1-based 位置的完整突变序列。

    位置超界或旧 AA 与 WT 不符时记录问题（不中断，交由调用方决定是否丢弃）。
    """
    seq = list(wt)
    problems: list[str] = []
    for tok in tokens:
        m = re.match(r"([A-Za-z])(\d+)([A-Za-z])", tok)
        if not m:
            continue
        old, pos_s, new = m.group(1), m.group(2), m.group(3)
        pos = int(pos_s)
        if pos < 1 or pos > len(seq):
            problems.append(f"{tok}: 位置 {pos} 超界(WT长{len(seq)})")
            continue
        if seq[pos - 1] != old:
            problems.append(f"{tok}: 旧AA {old}!=WT[{pos}]={seq[pos-1]}")
        if new not in _AA_SET:
            problems.append(f"{tok}: 新AA {new} 非标准")
            continue
        seq[pos - 1] = new
    return "".join(seq), problems


def load_dataset(xlsx: str, sheet: str, target: str, wt: str, site_col_hint: str = "") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """读 xlsx、按 Mut_ID 去重、把 token 组合转成完整突变序列并编码。

    Returns:
        (X, y, meta_rows) — X 为特征矩阵, y 为目标值, meta_rows 为逐样本审计信息。
    """
    df = pd.read_excel(xlsx, sheet_name=sheet)
    site_col = None
    if site_col_hint and site_col_hint in df.columns:
        site_col = site_col_hint
    if site_col is None:
        site_col = next((c for c in df.columns if "突变" in str(c) and "位点" in str(c)), None)
    if site_col is None:
        # 容错：取含「突变」且非目标列的第一列
        site_col = next((c for c in df.columns if "突变" in str(c) and str(c) != target), None)
    if site_col is None:
        # 容错：新数据集（Supplementary Material-2）用 ID 列承载突变 token
        site_col = next((c for c in df.columns if str(c).strip().lower() == "id"), None)
    if site_col is None:
        raise ValueError(f"在 sheet「{sheet}」中找不到『突变位点』列，现有列: {list(df.columns)}")

    id_col = None
    for cand in ("Mut_ID", "Mut ID", "编号", "ID"):
        if cand in df.columns:
            id_col = cand
            break

    featurizer = SequenceFeaturizer(features=["compo", "aaind"])

    rows: list[dict] = []
    for idx, row in df.iterrows():
        site = row.get(site_col)
        y = row.get(target)
        if site is None or pd.isna(site) or pd.isna(y):
            continue
        tokens = parse_tokens(str(site))
        if not tokens:
            continue
        mut_seq, problems = mut2seq(wt, tokens)
        if mut_seq is None:
            continue
        rows.append({
            "id": row.get(id_col) if id_col else None,
            "site": str(site),
            "y": float(y),
            "mut_seq": mut_seq,
            "tokens": tokens,
            "problems": problems,
        })
        if problems:
            print(f"[WARN] 样本 {str(site)}: {'; '.join(problems)}", file=sys.stderr)

    if not rows:
        raise ValueError(f"sheet「{sheet}」/ 目标列「{target}」没有可用标签行")

    # 按 Mut_ID 去重：同 ID 的活性取均值（处理复测重复）
    df_rows = pd.DataFrame(rows)
    if "id" in df_rows.columns:
        dup_ids = df_rows["id"].notna().sum()
        if dup_ids:
            # 分组取 y 均值，保留第一个 site/mut_seq
            df_rows = (
                df_rows.groupby("id", as_index=False)
                .agg({"site": "first", "y": "mean", "mut_seq": "first", "tokens": "first",
                      "problems": "first"})
            )

    X = np.vstack([featurizer.encode(s) for s in df_rows["mut_seq"]])
    y = df_rows["y"].astype(float).to_numpy()
    meta = df_rows.to_dict("records")
    return X, y, meta, featurizer


def evaluate(X: np.ndarray, y: np.ndarray, n_splits: int, random_state: int) -> dict:
    """KFold 交叉验证，输出 out-of-fold 指标。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from scipy.stats import spearmanr

    model = make_model()
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, X, y, cv=kf)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    try:
        sp, _ = spearmanr(y, pred)
        sp = float(sp) if np.isfinite(sp) else float("nan")
    except Exception:
        sp = float("nan")
    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "spearman": sp,
    }


def make_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(random_state=37, n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.9)


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="data/raw/Supplementary-Material-2.xlsx")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--target", default="activity")
    ap.add_argument("--site-col", default="", help="突变位点(旧AA+位点+新AA token)所在列名；默认自动识别『突变+位点』/『ID』")
    ap.add_argument("--wt", default=DEFAULT_WT, help="训练 WT 序列(默认已内嵌 360-aa)")
    ap.add_argument("--wt-file", default="", help="从文件读取训练 WT")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=37)
    args = ap.parse_args(argv)

    wt = args.wt
    if args.wt_file:
        wt = Path(args.wt_file).read_text(encoding="utf-8").strip()

    print(f"[1/4] 读取数据: {args.xlsx} | sheet「{args.sheet}」| 目标「{args.target}」")
    X, y, meta, featurizer = load_dataset(args.xlsx, args.sheet, args.target, wt, args.site_col)
    print(f"      样本数: {len(y)}  | 特征维度: {X.shape[1]}")

    print("[2/4] 交叉验证评估 ...")
    metrics = evaluate(X, y, args.n_splits, args.random_state)
    print(f"      R2={metrics['r2']:.3f}  MAE={metrics['mae']:.3f}  "
          f"RMSE={metrics['rmse']:.3f}  Spearman={metrics['spearman']:.3f}")

    print("[3/4] 全量训练并落盘 ...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = make_model()
    model.fit(X, y)

    import pickle
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"model": model, "featurizer_config": featurizer.config}, fh)
    with open(out / "featurizer_meta.json", "w", encoding="utf-8") as fh:
        json.dump(featurizer.config, fh, ensure_ascii=False, indent=2)
    with open(out / "wt.txt", "w", encoding="utf-8") as fh:
        fh.write(wt)
    with open(out / "target_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "xlsx": args.xlsx, "sheet": args.sheet, "target": args.target,
            "wt_len": len(wt), "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "cv": metrics,
        }, fh, ensure_ascii=False, indent=2)

    print(f"[4/4] 完成 → {out}/")
    print("      CV 指标: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                        if isinstance(v, float)))
    print("      该基线基于有限标签与测量噪声，R² 属入门级，仅供闭环参考。")


if __name__ == "__main__":
    main()
