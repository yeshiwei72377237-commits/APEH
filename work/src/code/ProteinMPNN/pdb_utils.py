#!/usr/bin/env python3
"""PDB 解析与氨基酸常量，供位置推荐与 ProteinMPNN 封装共用。

抽离自 mutation_position_recommender.py，避免两个脚本各自维护
一份氨基酸表和 PDB 解析逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HID": "H",
    "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Tien et al. theoretical maximum solvent-accessible areas, Å².
MAX_ASA_A2 = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}


@dataclass
class Residue:
    chain: str
    sequence_position: int
    pdb_resid: int
    insertion_code: str
    residue_name: str
    amino_acid: str

    @property
    def position(self) -> str:
        return f"{self.chain}{self.sequence_position}"


def parse_pdb_residues(pdb_path: Path, chain: str) -> list[Residue]:
    """以CA原子为准读取单链残基，并生成1起始链内位置。"""
    residues: list[Residue] = []
    seen: set[tuple[int, str]] = set()

    for line in pdb_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        atom_name = line[12:16].strip()
        line_chain = line[21].strip() or chain
        if atom_name != "CA" or line_chain != chain:
            continue
        try:
            pdb_resid = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (pdb_resid, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        residue_name = line[17:20].strip().upper()
        residues.append(Residue(
            chain=chain,
            sequence_position=len(residues) + 1,
            pdb_resid=pdb_resid,
            insertion_code=insertion_code,
            residue_name=residue_name,
            amino_acid=AA3_TO_1.get(residue_name, "X"),
        ))

    if not residues:
        raise ValueError(f"PDB中没有找到链{chain}的CA原子：{pdb_path}")
    return residues


def detect_first_chain(pdb_path: Path) -> str:
    """返回PDB中第一个含CA原子的链标识。"""
    for line in pdb_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21].strip()
        if chain:
            return chain
    raise ValueError(f"PDB中没有找到任何CA原子：{pdb_path}")
