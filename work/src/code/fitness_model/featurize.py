#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列 → 特征向量的确定性编码器（零外部依赖，仅需 numpy）。

设计对应 Model/ 的「AAindex + 氨基酸组成 + 二肽组成」思路，但把理化性质内嵌为
常量表，保证可复现、无 pySAR 依赖。训练(train.py)与打分(score.py)必须用同一
配置构造 SequenceFeaturizer，特征向量维度/布局才会一致。

特征组成（默认）：
  1. 氨基酸组成   (20 维)   —— 20 种标准 AA 在序列中的占比（X 及非标准残基忽略）
  2. 二肽组成     (400 维)  —— 相邻氨基酸对占比（可选，--dipeptide 开启，默认关）
  3. 理化性质     (5 尺度 × 均值+标准差 = 10 维) —— 疏水/极性/体积/电荷/分子量

默认关闭二肽组成（400 维对 223 条样本易过拟合），运行时先取「组成+理化」(30 维)。
"""

from __future__ import annotations

import json
import numpy as np

# 20 种标准氨基酸（固定的规范顺序）
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_LIST = list(AAS)

# ---- 内嵌的理化性质常量表（近似值，来自经典来源） ----
# 疏水性 Kyte-Doolittle (KYT820101)
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# 极性 Grantham
POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3,
    "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2,
    "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
# 侧链体积 (Å^3)
VOLUME = {
    "A": 67, "R": 148, "N": 114, "D": 111, "C": 86, "Q": 143, "E": 138, "G": 48,
    "H": 118, "I": 124, "L": 124, "K": 135, "M": 124, "F": 135, "P": 90, "S": 89,
    "T": 93, "W": 163, "Y": 141, "V": 105,
}
# 电荷（D/E= -1, R/K=+1, H=+0.5, 其余 0）
CHARGE = {"D": -1, "E": -1, "R": 1, "K": 1, "H": 0.5}
# 残基平均质量 (Da)
MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15, "Q": 146.14,
    "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23,
    "Y": 181.19, "V": 117.15,
}

# 内嵌属性表（顺序固定，形成稳定的特征子块）
AAINDEX_PROPERTIES = [
    ("hydrophobicity", HYDROPHOBICITY),
    ("polarity", POLARITY),
    ("volume", VOLUME),
    ("charge", CHARGE),
    ("mw", MW),
]

ALL_FEATURE_SETS = {"compo", "aaind", "dipep"}
# 每个属性贡献 2 维（mean + std）
PROPERTY_DIM = len(AAINDEX_PROPERTIES) * 2  # 10
COMPO_DIM = len(AA_LIST)                    # 20
DIPEP_DIM = len(AA_LIST) ** 2               # 400


def _sanitize(seq: str) -> str:
    """仅保留 20 种标准氨基酸（大写），去掉 X、'/' 及其他非标准残基。"""
    return "".join(c for c in (seq or "").upper() if c in AA_LIST)


class SequenceFeaturizer:
    """确定性序列特征编码器。

    Args:
        features: 特征子集，取值来自 ALL_FEATURE_SETS。顺序固定，影响向量布局。
    """

    def __init__(self, features: list[str] | None = None):
        self.features = list(features or ["compo", "aaind"])
        unknown = set(self.features) - ALL_FEATURE_SETS
        if unknown:
            raise ValueError(f"未知特征子集: {sorted(unknown)}")
        self.dim = self._compute_dim()

    @property
    def config(self) -> dict:
        """供写入 meta 的配置。"""
        return {"features": list(self.features)}

    def _compute_dim(self) -> int:
        d = 0
        if "compo" in self.features:
            d += COMPO_DIM
        if "aaind" in self.features:
            d += PROPERTY_DIM
        if "dipep" in self.features:
            d += DIPEP_DIM
        return d

    def encode(self, seq: str) -> np.ndarray:
        """把单条序列编码为长度固定 (=dim) 的 1D 向量。"""
        s = _sanitize(seq)
        if not s:
            s = "A" * len(AAS)  # 空序列时给一个退化向量，但现实中不会发生
        parts: list[np.ndarray] = []
        n = len(s)

        if "compo" in self.features:
            counts = np.bincount([AA_LIST.index(c) for c in s], minlength=len(AA_LIST))
            parts.append(counts / n)

        if "aaind" in self.features:
            # 对每个属性: [mean, std]，对每一列缺失属性填 0（如 charge 表缺 A→0）
            blk = []
            for _name, table in AAINDEX_PROPERTIES:
                vals = np.array([table.get(c, 0.0) for c in s], dtype=float)
                blk.append([float(vals.mean()), float(vals.std())])
            parts.append(np.array(blk, dtype=float).ravel())

        if "dipep" in self.features:
            if n < 2:
                dip = np.zeros(DIPEP_DIM, dtype=float)
            else:
                pairs = [AA_LIST.index(a) * len(AA_LIST) + AA_LIST.index(b)
                         for a, b in zip(s, s[1:])]
                dip = np.bincount(pairs, minlength=DIPEP_DIM) / (n - 1)
            parts.append(dip.astype(float))

        vec = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return vec.astype(float)

    def feature_names(self) -> list[str]:
        """返回每维的人类可读列名（用于调试/审计）。"""
        names: list[str] = []
        if "compo" in self.features:
            names += [f"comp_{a}" for a in AA_LIST]
        if "aaind" in self.features:
            for name, _t in AAINDEX_PROPERTIES:
                names += [f"{name}_mean", f"{name}_std"]
        if "dipep" in self.features:
            names += [f"dip_{a}{b}" for a in AA_LIST for b in AA_LIST]
        return names


def load_from_config(config: dict) -> SequenceFeaturizer:
    """从 meta/config 字典重建 featurizer（train.py 写入、score.py 读取）。"""
    return SequenceFeaturizer(features=config.get("features"))


def featurizer_to_config(f: SequenceFeaturizer) -> dict:
    return f.config


# ---- 简单自检：维度与确定性 ----
if __name__ == "__main__":
    f = SequenceFeaturizer(features=["compo", "aaind"])
    a = f.encode("ACDEFGHIKLMNPQRSTVWY")
    b = f.encode("ACDEFGHIKLMNPQRSTVWY")
    assert a.shape == (f.dim,) and np.allclose(a, b), "编码应确定且维度固定"
    fd = SequenceFeaturizer(features=["compo", "aaind", "dipep"])
    c = fd.encode("ACDEFGHIKLMNPQRSTVWY")
    assert c.shape == (fd.dim,), f"dims: {fd.dim}"
    print("featurize self-check OK | default dim:", f.dim,
          "| dipep dim:", fd.dim)
    print(json.dumps(f.config))
