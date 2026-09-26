#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用固定的 AAindex 代码列表重建 nested_cv.py 的全 AAindex 特征矩阵。

训练（train_final.py）与打分（score.py）共用本模块，保证部署模型看到的列与
训练时逐列一致。

编码规则与 nested_cv.py 完全相同——直接复用其
`load_aaindex1` / `get_mapping` / `clean_sequence` / `encode_aac` / `encode_dpc`，
唯一区别是代码列表由调用方给出（来自训练产物的 `valid_codes`）：
训练时若某条 AAindex 记录缺失会被跳过，打分时必须按同一份"成功代码"清单逐列重建，
否则列会整体错位。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nested_cv import (  # noqa: E402
    AA_ORDER,
    clean_sequence,
    encode_aac,
    encode_dpc,
    get_mapping,
    load_aaindex1,
)


def encode_with_codes(
    sequences: Sequence[str],
    valid_codes: Sequence[str],
    include_aac: bool,
    include_dpc: bool,
) -> Tuple[np.ndarray, List[str]]:
    """按 valid_codes 指定的 AAindex 记录逐位点编码，返回 (X, feature_names)。"""
    cleaned = [clean_sequence(value) for value in sequences]
    lengths = {len(sequence) for sequence in cleaned}
    if len(lengths) != 1:
        raise ValueError(
            "逐位点AAindex编码要求所有序列等长。长度分布："
            f"{sorted(lengths)}"
        )
    seq_len = lengths.pop()

    aa_to_int = {aa: index for index, aa in enumerate(AA_ORDER)}
    encoded_sequences = np.asarray(
        [[aa_to_int[aa] for aa in sequence] for sequence in cleaned],
        dtype=np.int16,
    )

    database = load_aaindex1()
    blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    for code in valid_codes:
        mapping = get_mapping(database, code)
        values = np.asarray([mapping[aa] for aa in AA_ORDER], dtype=np.float32)
        blocks.append(values[encoded_sequences])
        feature_names.extend(
            f"{code}_pos_{position}" for position in range(1, seq_len + 1)
        )

    if include_aac:
        block, names = encode_aac(cleaned)
        blocks.append(block)
        feature_names.extend(names)

    if include_dpc:
        block, names = encode_dpc(cleaned)
        blocks.append(block)
        feature_names.extend(names)

    if not blocks:
        raise RuntimeError("没有生成任何特征：valid_codes 为空且未启用 AAC/DPC。")

    matrix = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    return matrix, feature_names
