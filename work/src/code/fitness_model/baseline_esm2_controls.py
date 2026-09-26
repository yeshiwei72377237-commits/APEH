#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基线对比的**混淆控制实验** —— 回答「ESM-2 的 R²=0.572 是不是真的比本队强」。

为什么需要这个脚本
------------------
`baseline_esm2.py` 报出 ESM-2 冻结表征显著优于本队 30 维模型（R² 0.339 → 0.572）。
但这个数字**不能直接采信**，因为本数据集的 `activity` 标签与**突变个数**强相关：

- 数据集里 52/126 是单点突变，其余为 2~9 个位点的理性叠加组合；
- 标签均值随突变个数单调上升（1 个 → 1.10，9 个 → 2.77）；
- 相关系数 `pearson(突变个数, activity) = 0.743`。

一个能隐式数出「这条序列改了几个位点」的模型——ESM-2 表征的**到 WT 的欧氏距离**
天然就编码了这个量——就能白拿很高的 R²，**而完全不理解任何单点突变效应**。

因此本脚本做两个控制：

1. **控制 A（单变量下限）**：只用「突变个数」这一个整数特征跑同一套 10 折 CV。
   它给出的 R² 就是「不学任何序列信息、只数个数」的下限。任何模型若不显著高于它，
   就不能声称学到了序列层面的规律。

2. **控制 B（消除混淆）**：只在 **52 条单点突变**子集上重跑对比。子集内突变个数恒为 1，
   混淆项被消除，此时三个模型的差异才是**真正的单点效应预测能力**。

结论（本脚本的输出，见 results/baseline_esm2/controls.json）
------
- 控制 A：突变个数单独一个特征 → R² = 0.560，与本队 0.339 相比**更高**，
  与 ESM-2 的 0.556~0.572 **基本持平**。→ 全 126 条上的高 R² 主要是混淆项。
- 控制 B（n=52 单点）：本队 R² = **−0.527**（差于直接预测均值，Spearman +0.128 近于 0），
  ESM-2 R² = **+0.305**（Spearman +0.476）。
  → **本队模型在消除混淆后没有可用的单点效应预测能力；ESM-2 有。**

即：**「ESM-2 更强」这个结论要改写为「本队的 30 维位置盲特征基本只在拟合突变个数，
而 ESM-2 表征在单点层面确实携带了真实信号」**——后者才是站得住的表述。

用法
----
    python src/code/fitness_model/baseline_esm2_controls.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import load_dataset  # noqa: E402

XLSX = "data/raw/Supplementary-Material-2_2UWF.xlsx"
WT = HERE / "artifacts" / "wt.txt"
BASELINE_DIR = Path("results/baseline_esm2")


def gbr() -> GradientBoostingRegressor:
    """与本队 `make_model()`、基线 `esm2_gbr` 完全相同的回归头。"""
    return GradientBoostingRegressor(random_state=37, n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.9)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


def cv(model, X: np.ndarray, y: np.ndarray, seed: int = 37) -> tuple[float, float, np.ndarray]:
    """同一套折（与 train.evaluate / baseline_esm2.cv_metrics 一致）→ (R², Spearman, 预测)。"""
    from scipy.stats import spearmanr

    kf = KFold(n_splits=min(10, len(y)), shuffle=True, random_state=seed)
    p = cross_val_predict(model, X, y, cv=kf)
    sp = spearmanr(y, p)[0]
    return float(r2_score(y, p)), (float(sp) if np.isfinite(sp) else float("nan")), p


def main() -> None:
    from scipy.stats import spearmanr

    wt = WT.read_text(encoding="utf-8").strip()
    X_team, y, meta, _ = load_dataset(XLSX, "Sheet1", "activity", wt)
    emb = np.load(BASELINE_DIR / "emb_esm2_t33_650M_UR50D.npy")
    oof = pd.read_csv(BASELINE_DIR / "oof_predictions.csv")
    zs = oof["esm2_zeroshot"].to_numpy()
    assert len(zs) == len(y) and len(emb) == len(y)

    nmut = np.array([len(m["tokens"]) for m in meta], dtype=float)
    sel = nmut == 1

    out: dict = {
        "n_total": int(len(y)),
        "n_single": int(sel.sum()),
        "mutation_count_distribution": {
            str(int(k)): int(v) for k, v in zip(*np.unique(nmut, return_counts=True))
        },
        "confound": {
            "pearson_nmut_vs_activity": float(np.corrcoef(nmut, y)[0, 1]),
            "spearman_nmut_vs_activity": float(spearmanr(nmut, y)[0]),
            "mean_activity_by_nmut": {
                str(int(k)): float(y[nmut == k].mean()) for k in np.unique(nmut)
            },
        },
    }

    print("突变个数分布:", out["mutation_count_distribution"])
    print("activity 均值 按突变个数:",
          {k: round(v, 2) for k, v in out["confound"]["mean_activity_by_nmut"].items()})
    print("迷惑项强度: pearson(nmut, y)=%.3f  spearman=%.3f"
          % (out["confound"]["pearson_nmut_vs_activity"],
             out["confound"]["spearman_nmut_vs_activity"]))

    # ---- 控制 A：只用突变个数 ----
    print("\n=== 控制 A：仅「突变个数」一个特征（n=126）===")
    out["control_A_nmut_only"] = {}
    for name, model in [("gbr", gbr()), ("ridge", ridge())]:
        r, s, _ = cv(model, nmut.reshape(-1, 1), y)
        out["control_A_nmut_only"][name] = {"r2": r, "spearman": s}
        print(f"  nmut_only[{name:5s}]  R²={r:+.3f}  Spearman={s:+.3f}")

    # ---- 控制 B：单点突变子集 ----
    print(f"\n=== 控制 B：单点突变子集（n={int(sel.sum())}，突变个数恒定，混淆已消除）===")
    out["control_B_single_mutants"] = {
        "n": int(sel.sum()),
        "y_std": float(y[sel].std()),
        "y_var_share_of_total": float(y[sel].var() / y.var()),
    }
    for nm, X in [("team_30d", X_team[sel]), ("esm2_1280d", emb[sel])]:
        for mname, model in [("gbr", gbr()), ("ridge", ridge())]:
            r, s, _ = cv(model, X, y[sel])
            out["control_B_single_mutants"][f"{nm}__{mname}"] = {"r2": r, "spearman": s}
            print(f"  {nm:12s} {mname:5s}  R²={r:+.3f}  Spearman={s:+.3f}")
    zs_sp = float(spearmanr(y[sel], zs[sel])[0])
    out["control_B_single_mutants"]["esm2_zeroshot"] = {"spearman": zs_sp}
    print(f"  {'esm2_zeroshot':12s}        Spearman={zs_sp:+.3f}")

    # ---- 对照：全 126 条 ----
    print("\n=== 对照：全 126 条（含混淆项）===")
    out["control_C_all_126"] = {}
    for nm, X in [("team_30d", X_team), ("esm2_1280d", emb)]:
        r, s, _ = cv(gbr(), X, y)
        out["control_C_all_126"][nm] = {"r2": r, "spearman": s}
        print(f"  {nm:12s} gbr    R²={r:+.3f}  Spearman={s:+.3f}")
    for nm, X in [("team_30d+nmut", np.hstack([X_team, nmut.reshape(-1, 1)]))]:
        r, s, _ = cv(gbr(), X, y)
        out["control_C_all_126"][nm] = {"r2": r, "spearman": s}
        print(f"  {nm:12s} gbr    R²={r:+.3f}  Spearman={s:+.3f}")

    out["conclusion"] = (
        "混淆项（突变个数）单独即达 R²=0.560，与本队 0.339 相比更高、与 ESM-2 的 0.556~0.572 持平，"
        "故全 126 条上的排名主要反映「谁更能数出突变个数」。在突变个数恒定的 52 条单点突变上，"
        "本队模型 R²=-0.527（差于均值基线）、Spearman=+0.128；ESM-2 R²=+0.305、Spearman=+0.476。"
        "结论：本队 30 维位置盲特征基本只在拟合突变个数，ESM-2 表征在单点层面确有真实信号。"
    )
    out["reproduce"] = "python src/code/fitness_model/baseline_esm2_controls.py"
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    dst = BASELINE_DIR / "controls.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")


if __name__ == "__main__":
    main()
