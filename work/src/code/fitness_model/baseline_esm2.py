#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ESM-2 表征基线 —— 与下游 fitness 模型在同一套交叉验证折上对比。

背景
----
评审表 C.2 银牌第 3 条要求「AI 模型有明确的评估指标与**基线对比**（基线须为近三年
主流方法）」。本脚本提供该对比：把 ESM-2（Lin et al., Science 2023）作为**近三年主流
的蛋白语言模型表征**，在**同一批 126 条训练标签、同一套 10 折交叉验证折**上评估。

对比的三组
----------
| 名称 | 特征 | 回归头 |
|---|---|---|
| `team`   | 本队 30 维手工特征（20 AA 组成 + 5 理化尺度 mean/std） | GradientBoostingRegressor |
| `esm2_ridge` | ESM-2 末层隐状态**均值池化**（1280 维） | RidgeCV（线性探针，冻结表征的标准协议） |
| `esm2_gbr`   | 同上 | GradientBoostingRegressor（**与本队同一个头**） |

`esm2_ridge` 与 `esm2_gbr` 一起看，可以区分「特征差异」与「回归头差异」各贡献了多少。
`team` 组是**自校验**：它应当复现 `artifacts/target_meta.json` 里的 R²=0.3387 /
Spearman=0.6543；复现不了说明本脚本的折与主流程不一致，其余数字也不可信。

公平性说明（评审表 C.2 第 3 条提示要求注明）
--------------------------------------------
- 相同的 126 条样本、相同的目标列 `activity`、相同的 `KFold(n_splits=10, shuffle=True,
  random_state=37)`，出 out-of-fold 预测后统一算指标；
- 标准化（StandardScaler）与 RidgeCV 的 alpha 选择都在 `Pipeline` 内，**只在训练折上拟合**，
  不跨折泄漏；
- 本脚本是**本队自行复现**，不是 ESM-2 原作者的评测代码，也未沿用其超参数；
  只用其**公开预训练权重**做冻结表征。
- ESM-2 用的是**均值池化**的冻结表征，未做微调、未做逐残基突变效应打分。

用法
----
    # 首次运行会下载权重（650M 约 2.5 GB，缓存在 ~/.cache/torch/hub/checkpoints/）
    python src/code/fitness_model/baseline_esm2.py

    # 换更小的档（如实披露用哪一档即可）
    python src/code/fitness_model/baseline_esm2.py --esm-model esm2_t30_150M_UR50D
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import load_dataset, make_model  # noqa: E402

DEFAULT_XLSX = "data/raw/Supplementary-Material-2_2UWF.xlsx"
DEFAULT_WT = HERE / "artifacts" / "wt.txt"

# fair-esm 的 ESM-2 权重名（近三年主流档）
ESM2_CHOICES = {
    "esm2_t33_650M_UR50D": "650M",
    "esm2_t30_150M_UR50D": "150M",
    "esm2_t12_35M_UR50D": "35M",
    "esm2_t6_8M_UR50D": "8M",
}


def embed_esm2(seqs: list[str], model_name: str, batch_size: int, device: str,
               cache: Path | None) -> np.ndarray:
    """把序列池化成 ESM-2 的均值池化表征（冻结，不微调）。"""
    if cache is not None and cache.is_file():
        arr = np.load(cache)
        if arr.shape[0] == len(seqs):
            print(f"[2/4] 命中表征缓存 {cache} → {arr.shape}")
            return arr

    import esm
    import torch

    t0 = time.time()
    print(f"[2/4] 加载 ESM-2 权重 {model_name}（首次运行需下载）...", flush=True)
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    n_layers = model.num_layers
    print(f"      {model_name}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M 参数, "
          f"{n_layers} 层, device={device}  ({time.time()-t0:.0f}s)")

    converter = alphabet.get_batch_converter()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            _, _, tokens = converter([(f"s{j}", s) for j, s in enumerate(chunk)])
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[n_layers], return_contacts=False)["representations"][n_layers]
            # tokens 含 BOS/EOS：逐条按真实长度取 [1:len+1] 后均值池化
            for j, s in enumerate(chunk):
                out.append(reps[j, 1:len(s) + 1].mean(0).cpu().numpy())
            if (i // batch_size) % 5 == 0:
                print(f"      已编码 {min(i+batch_size, len(seqs))}/{len(seqs)}", flush=True)

    arr = np.vstack(out).astype(np.float32)
    print(f"      表征矩阵 {arr.shape}  ({time.time()-t0:.0f}s)")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
        print(f"      已缓存 → {cache}")
    return arr


def zeroshot_masked_marginal(meta: list[dict], wt: str, model_name: str, device: str,
                             batch_size: int, cache: Path | None) -> np.ndarray:
    """ESM-2 零样本「掩盖边际」突变打分（Meier et al. 2021 协议，ESM-2 论文沿用）。

    对每个突变 token `X{pos}Y`：把 WT 第 pos 位换成 `<mask>`，读该位点对 20 种标准氨基酸的
    log 概率，取 delta = logP(Y) − logP(X)；一条样本的分数 = 各 token delta 之和
    （多突变下的加性近似，与 ESM 官方 predict.py 的做法一致）。

    掩盖上下文只取决于 pos、与样本无关，故按**唯一位置**做前向再聚合（唯一位置数远少于样本数）。

    返回每条样本的分数。**该分数不经过任何训练**，因此不与 team / esm2_* 组共用交叉验证折，
    只报 Spearman（排序一致性）。
    """
    import re as _re

    import esm
    import torch
    import torch.nn.functional as F

    AA = "ACDEFGHIKLMNPQRSTVWY"
    tok_re = _re.compile(r"([A-Za-z])(\d+)([A-Za-z])")
    jobs: list[tuple[int, int, str, str]] = []
    for i, m in enumerate(meta):
        for tok in m["tokens"]:
            mm = tok_re.match(str(tok))
            if mm:
                jobs.append((i, int(mm.group(2)), mm.group(1).upper(), mm.group(3).upper()))
    positions = sorted({p for _, p, _, _ in jobs})
    print(f"      零样本：{len(jobs)} 个突变 token → {len(positions)} 个唯一位点")

    lp: dict[int, np.ndarray] = {}
    if cache is not None and cache.is_file():
        lp = {int(k): v for k, v in np.load(cache, allow_pickle=True).item().items()}
        print(f"      命中零样本缓存 {cache}")
    if not lp:
        model, alphabet = getattr(esm.pretrained, model_name)()
        model = model.to(device).eval()
        converter = alphabet.get_batch_converter()
        aa_idx = torch.tensor([alphabet.get_idx(a) for a in AA], device=device)
        with torch.no_grad():
            for i in range(0, len(positions), batch_size):
                chunk = positions[i:i + batch_size]
                data = [(f"p{p}", wt[:p - 1] + "<mask>" + wt[p:]) for p in chunk]
                _, _, tokens = converter(data)
                logits = model(tokens.to(device), repr_layers=[])["logits"]
                for j, p in enumerate(chunk):
                    # tokens 含 BOS，故 1-based 残基 p 对应 token 下标 p
                    logp = F.log_softmax(logits[j, p].float(), dim=-1)
                    lp[p] = logp[aa_idx].cpu().numpy().astype(np.float64)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, lp)
            print(f"      已缓存 → {cache}")

    a2i = {a: k for k, a in enumerate(AA)}
    scores = np.zeros(len(meta), dtype=np.float64)
    skipped = 0
    for i, p, old, new in jobs:
        if new not in a2i or old not in a2i or p not in lp:
            skipped += 1
            continue
        scores[i] += float(lp[p][a2i[new]] - lp[p][a2i[old]])
    if skipped:
        print(f"      [WARN] {skipped} 个 token 无法打分（非标准 AA 或位点缺上下文）")
    return scores


def cv_metrics(model, X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[dict, np.ndarray]:
    """与 train.evaluate() 完全一致的折与指标，另外把 out-of-fold 预测返回。"""
    from scipy.stats import spearmanr

    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=kf)
    spearman = float(spearmanr(y, pred)[0])
    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "spearman": spearman if np.isfinite(spearman) else float("nan"),
    }, pred


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ESM-2 表征基线（同折对比下游 fitness 模型）")
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--target", default="activity")
    ap.add_argument("--wt-file", default=str(DEFAULT_WT))
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D", choices=sorted(ESM2_CHOICES))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/baseline_esm2")
    ap.add_argument("--no-cache", action="store_true", help="不用/不写表征缓存")
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退 CPU", file=sys.stderr)
        args.device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 数据：复用主流程的 load_dataset，保证样本与去重口径完全一致 ----
    wt = Path(args.wt_file).read_text(encoding="utf-8").strip()
    print(f"[1/4] 读取 {args.xlsx} | 目标「{args.target}」| WT {len(wt)} aa")
    team_X, y, meta, _ = load_dataset(args.xlsx, args.sheet, args.target, wt)
    seqs = [m["mut_seq"] for m in meta]
    assert len(set(len(s) for s in seqs)) == 1, "候选链长不一致"
    print(f"      {len(seqs)} 条样本，链长 {len(seqs[0])} aa")

    # ---- 2. ESM-2 表征 ----
    cache = None if args.no_cache else out_dir / f"emb_{args.esm_model}.npy"
    emb = embed_esm2(seqs, args.esm_model, args.batch_size, args.device, cache)

    # ---- 3. 同一套折上评估三组 ----
    print("[3/4] 交叉验证（KFold n_splits=%d, shuffle=True, random_state=%d）" % (args.n_splits, args.seed))
    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    for name, X, model in [
        ("team", team_X, make_model()),
        ("esm2_ridge", emb, make_pipeline(StandardScaler(),
                                          RidgeCV(alphas=np.logspace(-2, 4, 13)))),
        ("esm2_gbr", emb, make_pipeline(StandardScaler(),
                                        GradientBoostingRegressor(random_state=37, n_estimators=300,
                                                                  max_depth=3, learning_rate=0.05,
                                                                  subsample=0.9))),
    ]:
        m, p = cv_metrics(model, X, y, args.n_splits, args.seed)
        results[name] = m
        preds[name] = p
        print(f"      {name:12s} R²={m['r2']:+.3f}  MAE={m['mae']:.3f}  "
              f"RMSE={m['rmse']:.3f}  Spearman={m['spearman']:+.3f}")

    # ---- 3b. 零样本（不做任何训练）基线 ----
    from scipy.stats import spearmanr

    zs_cache = None if args.no_cache else out_dir / f"zeroshot_{args.esm_model}.npy"
    zs = zeroshot_masked_marginal(meta, wt, args.esm_model, args.device, args.batch_size, zs_cache)
    zs_sp = float(spearmanr(y, zs)[0])
    zs_pear = float(np.corrcoef(y, zs)[0, 1])
    print(f"      {'esm2_zeroshot':12s} Spearman={zs_sp:+.3f}  Pearson={zs_pear:+.3f}  "
          f"（无训练，故不占交叉验证折）")

    # ---- 4. 落盘 ----
    ref = json.loads((HERE / "artifacts" / "target_meta.json").read_text(encoding="utf-8"))["cv"]
    same = abs(results["team"]["r2"] - ref["r2"]) < 1e-6
    print(f"[4/4] 自校验：本脚本 team 组 R²={results['team']['r2']:.6f} "
          f"vs artifacts/target_meta.json R²={ref['r2']:.6f} → {'一致 ✓' if same else '不一致 ✗'}")

    summary = {
        "esm_model": args.esm_model,
        "esm_size": ESM2_CHOICES[args.esm_model],
        "pooling": "mean over residues (frozen, no fine-tuning)",
        "xlsx": args.xlsx,
        "target": args.target,
        "n_samples": len(y),
        "cv": {"n_splits": args.n_splits, "shuffle": True, "random_state": args.seed},
        "results": results,
        "zeroshot": {
            "protocol": "masked-marginal (Meier et al. 2021)：WT 逐位 <mask> 后取 logP(新AA)−logP(旧AA)，各 token 求和",
            "training": "无（零样本，未使用任何标签）",
            "spearman": zs_sp,
            "pearson": zs_pear,
            "note": "无训练，故不与 results 各组共用交叉验证折，仅由全 126 条直接算排序一致性",
        },
        "self_check_team_vs_target_meta": {
            "script_team_r2": results["team"]["r2"],
            "target_meta_r2": ref["r2"],
            "match": bool(same),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame({"id": [m.get("id") for m in meta],
                       "site": [m["site"] for m in meta],
                       "y_true": y,
                       "esm2_zeroshot": zs,
                       **{f"pred_{k}": v for k, v in preds.items()}})
    df.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"      已写出 {out_dir}/metrics.json 与 oof_predictions.csv")


if __name__ == "__main__":
    main()
