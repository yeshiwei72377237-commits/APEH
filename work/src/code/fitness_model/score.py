#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用训练好的 fitness 模型给 ProteinMPNN 候选打分，并把 PDB 残基号翻译成序列位置 token。

输入 run_dir（ProteinMPNN 单运行目录）：
    <name>.pdb              目录内唯一 PDB（用于定位名称）
    seqs/<name>.fa          首条 record 为参考/WT 序列
    resid_mapping.json      mapping[chain][i] = PDB残基号（下标 i ↔ 链内序号 i+1）
    audited_candidates.csv  （缺省回落 candidates.csv）
输出：
    <dir>/scored_candidates.csv  增加 Fitness_prediction / Mutant(序列位置) / score_invalid

用法：
    python score.py --dir <run_dir> --artifacts src/code/fitness_model/artifacts --chain A

蛋白特异约束：模型只对与训练 WT 同源、且链长一致的蛋白有效；不符的候选会标
score_invalid=1 且不产出分数。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from featurize import SequenceFeaturizer
from safety_gate import is_blocked, normalize

_TOKEN_RE = re.compile(r"^(?P<chain>[A-Za-z0-9]):(?P<old>[A-Za-z])(?P<resid>\d+)(?P<new>[A-Za-z])$")


def read_fasta_record0(path: Path) -> str:
    """读 FASTA 首条 record 的序列（参考/WT 序列）。"""
    lines = []
    started = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            if started:
                lines.append(line)
    return "".join(lines)


def find_pdb(directory: Path) -> Path:
    pdbs = sorted(directory.glob("*.pdb"))
    if not pdbs:
        # 容错：退而求其次找任何 pdb
        pdbs = sorted(directory.rglob("*.pdb"))
    if len(pdbs) != 1:
        raise RuntimeError(f"运行目录应含唯一 *.pdb，实际 {len(pdbs)} 个: {[p.name for p in pdbs]}")
    return pdbs[0]


def build_pos_by_resid(mapping: dict, chain: str) -> dict:
    """把 mapping[chain]（下标 i ↔ 链内序号 i+1 的 PDB残基号）反建成 {PDB残基号: 链内序号}。"""
    res = {}
    for i, resid in enumerate(mapping.get(chain, [])):
        try:
            res[int(resid)] = i + 1
        except (TypeError, ValueError):
            continue
    return res


def translate_mutations(mutations: str, mapping_by_chain: dict) -> tuple[str, str]:
    """把 PDB 残基号 mutations → 无链「序列位置」token。

    Input:  'A:E33I;A:K65R'    Output: ('E<seqpos>I,K<seqpos>R', 翻译失败的 token 列表)
    注：输出用逗号分隔，以匹配 select_top_mutants.py 的 canonicalize 按逗号切分。
    返回 (seqpos_mutant, failures)。
    """
    out, failures = [], []
    for chunk in str(mutations or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _TOKEN_RE.match(chunk)
        if not m:
            # 记录 LENGTH_MISMATCH / CHAIN_COUNT_MISMATCH 等异常值，不强行翻译
            failures.append(chunk)
            continue
        chain = m.group("chain")
        pos_by_resid = mapping_by_chain.get(chain, {})
        resid = int(m.group("resid"))
        seq_pos = pos_by_resid.get(resid)
        if seq_pos is None:
            # 找不到映射：退化为直接把数值当作序列位置（尽力而为，并标记）
            seq_pos = resid
        out.append(f"{m.group('old')}{seq_pos}{m.group('new')}")
    return ",".join(out), failures


def load_candidates(candidates_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(candidates_csv)
    # 统一列名：audited 用 best_mpnn_score / unique_candidate_id；candidates 用 mpnn_score / candidate_id
    rename = {}
    for src, dst in [("best_mpnn_score", "mpnn_score"), ("mpnn_score", "mpnn_score"),
                     ("unique_candidate_id", "source_candidate_id"),
                     ("candidate_id", "source_candidate_id"),
                     ("best_sample", "sample")]:
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    return df


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="ProteinMPNN 单运行目录")
    ap.add_argument("--artifacts", required=True, help="train.py 产出的 artifacts 目录")
    ap.add_argument("--chain", default="A", help="设计链名（用于映射）")
    ap.add_argument("--candidates", default="", help="候选 CSV；默认寻找 audited_candidates.csv")
    ap.add_argument("--output", default="", help="输出 CSV 路径；默认 <dir>/scored_candidates.csv")
    ap.add_argument("--wt", default="", help="覆盖训练 WT 校验序列（默认取 artifacts/wt.txt）")
    args = ap.parse_args(argv)

    run_dir = Path(args.dir).resolve()
    art = Path(args.artifacts).resolve()

    # ---- 加载 artifacts ----
    with open(art / "model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model = bundle["model"]
    train_wt = args.wt or (art / "wt.txt").read_text(encoding="utf-8").strip()
    with open(art / "featurizer_meta.json", encoding="utf-8") as fh:
        meta_config = json.load(fh)
    featurizer = SequenceFeaturizer(features=meta_config["features"])

    # ---- 输入 ----
    pdb = find_pdb(run_dir)
    name = pdb.stem
    fasta = run_dir / "seqs" / f"{name}.fa"
    ref_wt = read_fasta_record0(fasta) if fasta.is_file() else ""
    mapping_path = run_dir / "resid_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {"mapping": {}}
    mapping_by_chain = {c: build_pos_by_resid(mapping.get("mapping", {}), c) for c in
                        (set(mapping.get("mapping", {}).keys()) | {args.chain})}

    cand_csv = Path(args.candidates) if args.candidates else None
    if cand_csv is None:
        for fn in ("audited_candidates.csv", "candidates.csv"):
            p = run_dir / fn
            if p.is_file():
                cand_csv = p
                break
    if cand_csv is None:
        raise RuntimeError(f"在 {run_dir} 找不到 audited_candidates.csv 或 candidates.csv")
    df = load_candidates(cand_csv)

    out_path = Path(args.output) if args.output else run_dir / "scored_candidates.csv"

    records = []
    for _, row in df.iterrows():
        seq = str(row.get("sequence", "") or "")
        sanitized = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
        # 人工安全门控：review_status 命中 BLOCKED_STATUSES 的候选禁止进入评分（评审 P0）。
        # 词表见 safety_gate.py —— 必须同时覆盖人工复核的 HOLD/BLOCK 与自动审核写出的
        # HOLD_FOR_STRUCTURE_REVIEW。历史缺陷：此处曾只判断 {"HOLD","BLOCK"}，
        # 与 audit_candidates_csv 写出的 HOLD_FOR_STRUCTURE_REVIEW 字符串不一致，导致门控从未触发。
        # score_invalid 取值：0=有效 , 1=蛋白/长度不符 , 2=人工/自动审核标记禁止评分。
        review_status = normalize(row.get("review_status", ""))
        invalid = 0
        if is_blocked(review_status):
            invalid = 2
            fitness = float("nan")
            print(f"[HOLD] 候选 {row.get('source_candidate_id', '?')} 的 review_status="
                  f"{review_status}，属安全门控拦截项，已跳过评分（score_invalid=2）。",
                  file=sys.stderr)
        elif len(sanitized) != len(train_wt):
            invalid = 1
            fitness = float("nan")
        else:
            try:
                vec = featurizer.encode(sanitized).reshape(1, -1)
                fitness = float(model.predict(vec)[0])
            except Exception as e:
                invalid = 1
                fitness = float("nan")
                print(f"[WARN] 候选 {row.get('source_candidate_id', '?')} 预测失败: {e}",
                      file=sys.stderr)

        seqpos_mutant, fails = translate_mutations(str(row.get("mutations", "") or ""), mapping_by_chain)
        if fails:
            print(f"[WARN] 候选 {row.get('source_candidate_id', '?')} 无法翻译的突变 token: {fails}",
                  file=sys.stderr)

        records.append({
            "Mutant": seqpos_mutant,
            "Sequence": seq,
            "Fitness_prediction": fitness,
            "mpnn_score": row.get("mpnn_score", ""),
            "source_candidate_id": row.get("source_candidate_id", ""),
            "review_status": row.get("review_status", ""),
            "mutations_pdb": str(row.get("mutations", "") or ""),
            "score_invalid": invalid,
            "seq_len": len(sanitized),
        })

    out = pd.DataFrame(records)
    out.to_csv(out_path, index=False, float_format="%.6f")
    n_ok = int((out["score_invalid"] == 0).sum())
    n_hold = int((out["score_invalid"] == 2).sum()) if "score_invalid" in out else 0
    n_bad = int((out["score_invalid"] == 1).sum()) if "score_invalid" in out else 0
    print(f"[OK] 已打分 {len(out)} 条 → {out_path}")
    print(f"     有效(score_invalid=0): {n_ok}  | 蛋白/长度不符(score_invalid=1): {n_bad}"
          f"  | 人工HOLD/BLOCK(2): {n_hold}")
    if n_bad:
        ident_note = f"  训练WT长{len(train_wt)}、候选长度不均等; 请确认 MPNN 输入是与训练数据同源的蛋白。"
        print(f"     {ident_note}", file=sys.stderr)


if __name__ == "__main__":
    main()
