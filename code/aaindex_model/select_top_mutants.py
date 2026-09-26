#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3.3 — Adds a final cleanup step before saving CSV:
- If any selected combos have length 1 or 2, remove them and randomly replace with length in 4–7.
- If any selected combos have length > 12, remove them and randomly replace with length in 4–12.
(Replacement draws from remaining eligible pool with Fitness_prediction > min_pred and not already selected.)

Also keeps all v3.2 logic:
- Focus lengths 4–7; ensure each appears if available (> min_pred).
- Must include each file's top-prediction group:
    * If top includes 4–7: include up to --max_file_top_focus from that top group.
    * Otherwise include exactly 1 (even if length is 1 or 2).
- After forced picks, length 1–2 are excluded from normal selection.
- Other-length share ≤ other_max_pct (forced may exceed), with at least other_min if available.
- Priority positions format: R115,E145 (position-level).
- Diversity via Jaccard; per-value cap for non-forced selections.
"""
import argparse, os, glob, ast, math, random
import pandas as pd
import numpy as np
from collections import Counter

from safety_gate import BLOCKED_STATUSES, normalize

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="S7_mut_*.csv")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--per_value_cap", type=int, default=5)
    ap.add_argument("--similarity_threshold", type=float, default=0.8)
    ap.add_argument("--focus_lengths", default="4,5,6,7")
    ap.add_argument("--other_max_pct", type=float, default=0.10)
    ap.add_argument("--other_min", type=int, default=1)
    ap.add_argument("--priority_positions", default="")
    ap.add_argument("--min_pred", type=float, default=1.0)
    ap.add_argument("--max_file_top_focus", type=int, default=5)
    ap.add_argument("--random_seed", type=int, default=42, help="Seed for replacement randomness")
    return ap.parse_args()

def parse_token(token: str):
    s = str(token).strip().upper()
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    orig = s[:i] if i>0 else ""
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    pos = s[i:j]
    new = s[j:] if j < len(s) else ""
    try:
        pos_i = int(pos) if pos else None
    except Exception:
        pos_i = None
    return orig, pos_i, new

def canonicalize_mutant_and_sets(mutant_str):
    try:
        obj = ast.literal_eval(str(mutant_str))
        if isinstance(obj, (list, tuple)):
            items = [str(x).strip() for x in obj]
        else:
            items = [str(obj).strip()]
    except Exception:
        s = str(mutant_str).strip().strip("()[]")
        items = [x.strip().strip("'").strip('"') for x in s.split(",") if x.strip()]
    items_sorted = sorted(items)
    tok_set = set(items_sorted)
    pos_pairs = set()
    for t in items_sorted:
        orig, pos, new = parse_token(t)
        if orig and pos is not None:
            pos_pairs.add((orig, pos))
    return items_sorted, tok_set, pos_pairs

def parse_priority_positions(s: str):
    out = set()
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        orig, pos, new = parse_token(raw)
        if orig and pos is not None:
            out.add((orig, pos))
    return out

def load_all(input_dir, pattern):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            try:
                df = pd.read_excel(p)
            except Exception:
                continue
        colmap = {c.lower(): c for c in df.columns}
        mcol = None
        for cand in ["mutant","mutants","mutant_combo","combo"]:
            if cand in colmap:
                mcol = colmap[cand]; break
        if mcol is None:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            mcol = obj_cols[0] if obj_cols else df.columns[0]
        scol = None
        for cand in ["seq","sequence","seq_str"]:
            if cand in colmap:
                scol = colmap[cand]; break
        if scol is None:
            for c in df.columns:
                if df[c].dtype == object and df[c].astype(str).str.len().median() > 50:
                    scol = c; break
        if scol is None:
            scol = df.columns[1] if df.shape[1] > 1 else mcol
        pcol = None
        for cand in ["fitness_pred","fitness_prediction","pred","prediction","y_pred","score"]:
            if cand in colmap:
                pcol = colmap[cand]; break
        if pcol is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            pcol = num_cols[-1] if num_cols else df.columns[-1]
        sub = df[[mcol, scol, pcol]].copy()
        sub.columns = ["Mutant","Sequence","Fitness_prediction"]
        sub["source_file"] = os.path.basename(p)
        # 保留评分/审核状态列，用于人工安全门控（评审 P0）：HOLD/BLOCK 与评分无效候选禁止入选
        for extra in ("score_invalid", "review_status"):
            key = colmap.get(extra)
            if key and key in df.columns:
                sub[extra] = df[key].astype(str).values
        frames.append(sub)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["Mutant","Sequence","Fitness_prediction","source_file"])

def select_top(df, args):
    df = df[df["Fitness_prediction"] > args.min_pred].copy()
    if df.empty:
        return pd.DataFrame(), "无符合预期的突变位点组合"

    # ----- 安全门控（评审 P0）-----
    # review_status 命中 BLOCKED_STATUSES（人工 HOLD/BLOCK，或自动审核的
    # HOLD_FOR_STRUCTURE_REVIEW）、或评分无效（score_invalid!=0）的候选一律不得进入 TopN。
    # 此处显式排除并打印，而非依赖 NaN 比较的隐含行为，便于审计追溯。
    # 词表见 safety_gate.py：历史缺陷是此处只判断 {"HOLD","BLOCK"}，与自动审核写出的
    # HOLD_FOR_STRUCTURE_REVIEW 字符串不一致，导致门控从未触发。
    if "review_status" in df.columns:
        df["_rs_u"] = df["review_status"].map(normalize)
        _blocked = df["_rs_u"].isin(BLOCKED_STATUSES)
        n_blocked = int(_blocked.sum())
        if n_blocked:
            print(f"[BLOCK] 已排除 {n_blocked} 条被安全门控标记（{sorted(BLOCKED_STATUSES)}）"
                  f"的候选，禁止进入 TopN",
                  file=sys.stderr)
        df = df[~_blocked].drop(columns=["_rs_u"])
    if "score_invalid" in df.columns:
        df["_si_i"] = pd.to_numeric(df["score_invalid"], errors="coerce").fillna(0)
        n_invalid = int((df["_si_i"] != 0).sum())
        if n_invalid:
            print(f"[BLOCK] 已排除 {n_invalid} 条评分无效(score_invalid!=0)的候选", file=sys.stderr)
        df = df[df["_si_i"] == 0].drop(columns=["_si_i"])
    if df.empty:
        return pd.DataFrame(), "无符合预期的突变位点组合"

    df["Mutant_list"], df["Mutant_set"], df["Pos_pairs"] = zip(*df["Mutant"].astype(str).map(canonicalize_mutant_and_sets))
    df["mut_count"] = df["Mutant_list"].apply(len)

    focus_lengths = sorted({int(x.strip()) for x in str(args.focus_lengths).split(",") if x.strip()})
    focus_set = set(focus_lengths)
    mid = sum(focus_lengths)/len(focus_lengths) if focus_lengths else 5.5

    pri_set = parse_priority_positions(args.priority_positions)
    def count_priority_hits(pos_pairs):
        if not pri_set: return 0
        return sum(1 for p in pos_pairs if p in pri_set)
    df["pri_hits"] = df["Pos_pairs"].apply(count_priority_hits)
    df["is_focus_len"] = df["mut_count"].apply(lambda m: m in focus_set)
    df["len_closeness"] = - (df["mut_count"] - mid).abs()

    df = df.sort_values(
        ["Fitness_prediction","pri_hits","is_focus_len","len_closeness"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    # ----- Forced per-file top picks -----
    forced = []
    seen_forced_keys = set()
    def row_key(row):
        return (str(row["Mutant"]), str(row["Sequence"]), float(row["Fitness_prediction"]))

    for file_name, g in df.groupby("source_file", sort=False):
        if g.empty: continue
        max_pred = g["Fitness_prediction"].max()
        top_group = g[g["Fitness_prediction"] == max_pred].copy()
        top_group = top_group.sort_values(["pri_hits","is_focus_len","len_closeness"], ascending=[False, False, False])
        in_focus = top_group[top_group["is_focus_len"]]
        if not in_focus.empty:
            picks = in_focus.head(args.max_file_top_focus)
        else:
            picks = top_group.head(1)  # may be length 1–2; allowed as forced
        for _, r in picks.iterrows():
            k = row_key(r)
            if k not in seen_forced_keys:
                forced.append(r)
                seen_forced_keys.add(k)

    selected = []
    selected_sets = []
    selected_keys = set()
    count_focus_each = Counter()
    count_other = 0
    other_max = int(math.floor(args.other_max_pct * args.top_k))

    for r in forced:
        k = row_key(r)
        if k in selected_keys: 
            continue
        selected.append(r)
        selected_sets.append(r["Mutant_set"])
        selected_keys.add(k)
        if int(r["mut_count"]) in focus_set:
            count_focus_each[int(r["mut_count"])] += 1
        else:
            count_other += 1

    # Normal selection excludes length 1–2
    df_normal = df[df["mut_count"] >= 3].copy()

    remaining_slots = max(0, args.top_k - len(selected))
    value_counts = Counter()

    def violates_similarity(cand_set, thr):
        for s in selected_sets:
            inter = len(cand_set & s); union = len(cand_set | s)
            if union and inter/union > thr: return True
        return False

    def try_add(row, sim_thr, allow_other=True):
        nonlocal remaining_slots, count_other
        if remaining_slots <= 0: return False
        k = row_key(row)
        if k in selected_keys: return False
        pred = float(row["Fitness_prediction"])
        if value_counts[pred] >= args.per_value_cap: return False
        if violates_similarity(row["Mutant_set"], sim_thr): return False
        mlen = int(row["mut_count"])
        is_focus = (mlen in focus_set)
        if not is_focus:
            if not allow_other: return False
            if count_other >= other_max: return False
        selected.append(row)
        selected_sets.append(row["Mutant_set"])
        selected_keys.add(k)
        value_counts[pred] += 1
        if is_focus:
            count_focus_each[mlen] += 1
        else:
            count_other += 1
        remaining_slots -= 1
        return True

    sim_thr = args.similarity_threshold

    # Cover each focus length at least once (if available and not covered by forced)
    for L in focus_lengths:
        if sum(1 for x in selected if int(x["mut_count"]) == L) >= 1:
            continue
        poolL = df_normal[(df_normal["mut_count"] == L)]
        for _, r in poolL.iterrows():
            if try_add(r, sim_thr, allow_other=False):
                break

    # Ensure presence of "other" (excluding len 1–2)
    if args.other_min > 0:
        have_other = sum(1 for x in selected if int(x["mut_count"]) not in focus_set)
        need = max(0, args.other_min - have_other)
        if need > 0:
            others_pool = df_normal[~df_normal["is_focus_len"]]
            for _, r in others_pool.iterrows():
                if need <= 0: break
                if try_add(r, sim_thr, allow_other=True):
                    need -= 1

    # Fill remaining slots
    for _, r in df_normal.iterrows():
        if remaining_slots <= 0: break
        is_focus = (int(r["mut_count"]) in focus_set)
        if is_focus:
            try_add(r, sim_thr, allow_other=False)
        else:
            try_add(r, sim_thr, allow_other=True)

    # Relax similarity if still short
    if remaining_slots > 0:
        for thr in [min(0.85, sim_thr+0.05), min(0.9, sim_thr+0.1), 0.95, 0.99, 1.0]:
            if remaining_slots <= 0: break
            for _, r in df_normal.iterrows():
                if remaining_slots <= 0: break
                is_focus = (int(r["mut_count"]) in focus_set)
                if is_focus:
                    try_add(r, thr, allow_other=False)
                else:
                    try_add(r, thr, allow_other=True)

    # ------------- Final cleanup & random replacement -------------
    # Build DataFrame of selected, with metadata to support replacement
    sel_df = pd.DataFrame(selected).drop_duplicates(subset=["Mutant","Sequence","Fitness_prediction"]).reset_index(drop=True)
    if sel_df.empty:
        return sel_df, "无符合预期的突变位点组合"

    # Annotate mut_count for selected
    sel_df["mut_count"] = sel_df["Mutant"].astype(str).apply(lambda s: len(ast.literal_eval(s)) if s.startswith("(") or s.startswith("[") else (s.count(",")+1 if s else 0))

    rng = random.Random(args.random_seed)

    # Helper to get candidate pool for replacement with constraints
    def candidates_for_replacement(target_range):
        low, high = target_range
        pool = df[(df["mut_count"] >= low) & (df["mut_count"] <= high)].copy()
        # exclude already selected keys
        pool["_key"] = list(zip(pool["Mutant"].astype(str), pool["Sequence"].astype(str), pool["Fitness_prediction"].astype(float)))
        existing_keys = set(zip(sel_df["Mutant"].astype(str), sel_df["Sequence"].astype(str), sel_df["Fitness_prediction"].astype(float)))
        pool = pool[~pool["_key"].isin(existing_keys)]
        return pool

    # Remove and replace short (len 1–2), unless they were forced tops — but requirement says: if they appear in final CSV, delete & replace
    bad_short_idx = sel_df.index[sel_df["mut_count"] <= 2].tolist()
    if bad_short_idx:
        sel_df.drop(index=bad_short_idx, inplace=True)
        sel_df.reset_index(drop=True, inplace=True)
        need = max(0, args.top_k - len(sel_df))
        if need > 0:
            pool = candidates_for_replacement((4,7))
            if not pool.empty:
                picks = pool.sample(n=min(need, len(pool)), random_state=args.random_seed)
                add = picks[["Mutant","Sequence","Fitness_prediction"]]
                sel_df = pd.concat([sel_df, add], ignore_index=True)

    # Remove and replace too-long (>12)
    sel_df["mut_count"] = sel_df["Mutant"].astype(str).apply(lambda s: len(ast.literal_eval(s)) if s.startswith("(") or s.startswith("[") else (s.count(",")+1 if s else 0))
    bad_long_idx = sel_df.index[sel_df["mut_count"] > 12].tolist()
    if bad_long_idx:
        sel_df.drop(index=bad_long_idx, inplace=True)
        sel_df.reset_index(drop=True, inplace=True)
        need = max(0, args.top_k - len(sel_df))
        if need > 0:
            pool = candidates_for_replacement((4,12))
            if not pool.empty:
                picks = pool.sample(n=min(need, len(pool)), random_state=args.random_seed)
                add = picks[["Mutant","Sequence","Fitness_prediction"]]
                sel_df = pd.concat([sel_df, add], ignore_index=True)

    # Ensure exactly top_k rows if possible
    if len(sel_df) > args.top_k:
        sel_df = sel_df.head(args.top_k)

    # Keep only required columns
    keep_cols = ["Mutant","Sequence","Fitness_prediction"]
    # 可选保留审核/评分状态列，便于人工安全门控追溯（评审 P0）
    for extra in ("review_status", "score_invalid"):
        if extra in sel_df.columns:
            keep_cols.append(extra)
    sel_df = sel_df[keep_cols].reset_index(drop=True)
    return sel_df, None

def main():
    args = parse_args()
    df = load_all(args.input_dir, args.pattern)
    if df.empty:
        pd.DataFrame(columns=["Mutant","Sequence","Fitness_prediction"]).to_csv(args.output_csv, index=False)
        print("无符合预期的突变位点组合")
        return
    selected, msg = select_top(df, args)
    if selected is None or selected.empty:
        pd.DataFrame(columns=["Mutant","Sequence","Fitness_prediction"]).to_csv(args.output_csv, index=False)
        print("无符合预期的突变位点组合" if msg is None else msg)
    else:
        selected.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"已选择 {len(selected)} 个组合，输出到：{args.output_csv}")

if __name__ == "__main__":
    main()
