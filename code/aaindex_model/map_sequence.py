#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2UWF 构建体（356 aa）↔ S7 全长（360 aa）序列坐标映射。

背景（详见 src/code/fitness_model/2UWF_construct_mapping.md）：
    * S7 全长前体 360 aa；RCSB 2UWF 是 356 aa 成熟域构建体（催化域 + C 端 6×His）。
    * 两者用全局比对确定对应关系：构建体坐标 2..350 ↔ S7 坐标 12..360（1:1），
      构建体 1 号残基对应 S7 10 号；PDB 残基号 = S7 坐标（例如 A:T120K ↔ S7 120）。
    * 因此「按 PDB 编号书写的突变标签」可以 1:1 平移到 S7 坐标。

S7 活性模型在 360 aa / S7 编号下训练，而 ProteinMPNN 候选是 356 aa 构建体，
所以打分前必须把候选序列搬回 S7 坐标：用比对映射逐位点搬运，构建体覆盖不到的
S7 位点（N 端前导区等）保留 S7 野生型残基（候选不可能在那里突变）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def read_fasta_like(path: Path) -> str:
    """读取单行/多行纯序列文本（忽略空白与 FASTA 头）。"""
    seq_chars: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        seq_chars.append(line)
    return "".join(seq_chars).upper()


def wt_s7_360() -> str:
    return read_fasta_like(DATA / "wt_s7_360.txt")


def wt_construct_356() -> str:
    return read_fasta_like(DATA / "wt_2uwf_356.txt")


def build_alignment_map(
    construct: str | None = None,
    s7: str | None = None,
) -> Dict[int, int]:
    """返回 {构建体 0 基下标: S7 0 基下标} 的比对映射。"""
    from Bio.Align import PairwiseAligner

    construct = construct or wt_construct_356()
    s7 = s7 or wt_s7_360()

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(s7, construct)[0]
    mapping: Dict[int, int] = {}
    for (s_start, s_end), (c_start, c_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(min(s_end - s_start, c_end - c_start)):
            mapping[c_start + offset] = s_start + offset
    return mapping


def construct_to_s7(sequence: str, mapping: Dict[int, int] | None = None) -> str:
    """把 356 aa 构建体序列搬到 S7 360 aa 坐标；未覆盖位点保留 S7 野生型。"""
    if len(sequence) != len(wt_construct_356()):
        raise ValueError(
            f"候选序列长度 {len(sequence)} 与 2UWF 构建体 "
            f"{len(wt_construct_356())} 不一致，无法映射到 S7 坐标。"
        )
    mapping = mapping or build_alignment_map()
    target = list(wt_s7_360())
    for c_index, s_index in mapping.items():
        target[s_index] = sequence[c_index]
    return "".join(target)


def mutations_to_s7(mutations_pdb: str) -> List[str]:
    """把 PDB 编号的突变标签（A:T120K;D164K）原样保留为 S7 编号标签。

    由于 PDB 残基号 == S7 坐标，这里只做清洗与规范化，不做位移。
    """
    tokens: List[str] = []
    for raw in str(mutations_pdb or "").replace(",", ";").split(";"):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            token = token.split(":", 1)[1]
        tokens.append(token)
    return tokens


def _selftest() -> int:
    construct = wt_construct_356()
    s7 = wt_s7_360()
    mapping = build_alignment_map()
    mismatches: List[Tuple[int, int]] = []
    for c_index, s_index in sorted(mapping.items()):
        if construct[c_index] != s7[s_index]:
            mismatches.append((c_index, s_index))
    print(f"构建体长度 {len(construct)}  S7 长度 {len(s7)}")
    print(f"比对覆盖构建体 {len(mapping)}/{len(construct)} 个位点")
    print(f"构建体/S7 残基不一致位点（构建体下标, S7 下标）：{mismatches}")
    print("覆盖的 S7 坐标范围："
          f"{min(mapping.values())}..{max(mapping.values())}")
    print(f"construct_to_s7(构建体 WT) 结果长度：{len(construct_to_s7(construct))}")
    print(f"搬迁后与 S7 WT 差异：{sum(a != b for a, b in zip(construct_to_s7(construct), s7))} 个残基")
    print("PDB 120 号对应构建体下标：",
          [c for c, s in mapping.items() if s == 119])
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    raise SystemExit(_selftest())
