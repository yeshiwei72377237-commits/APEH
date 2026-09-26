#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_weights.py — ProteinMPNN 模型权重下载 + SHA-256 校验。

审计要求：模型权重作为「带哈希的下载步骤」，而不是悄悄塞进公开仓库的大二进制。
本脚本用于：复现 / 评审时校验权重完整性与归属，并给出获取方式。

说明（如实）：
- 这些 .pt 权重来自官方开源仓库 dauparas/ProteinMPNN（https://github.com/dauparas/ProteinMPNN），
  属于第三方模型权重，非本项目自训。官方仓库的 model_weights/ 目录即含这些文件。
- 本仓库内 `src/code/ProteinMPNN/ProteinMPNN-main/model_weights/` 已放置一份用于运行；
  如果你克隆仓库后发现权重缺失，用下面的命令获取，再跑本脚本校验 SHA-256。

用法：
    # 校验仓库内已存在的权重：
    python src/code/ProteinMPNN/download_weights.py --verify

    # 列出每个权重应有的 SHA-256 与来源，不做校验：
    python src/code/ProteinMPNN/download_weights.py --list

获取方式（官方）：
    git clone https://github.com/dauparas/ProteinMPNN
    # 再把 clone 下的 {vanilla,soluble,ca}_model_weights/*.pt 拷到
    # src/code/ProteinMPNN/ProteinMPNN-main/model_weights/ 对应子目录
"""
import argparse
import hashlib
import os
import sys

# 权重清单：子目录/文件名 -> 期望 SHA-256（本仓库内文件实测所得）
WEIGHTS = {
    "vanilla_model_weights/v_48_002.pt": "925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131",
    "vanilla_model_weights/v_48_010.pt": "db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce",
    "vanilla_model_weights/v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "vanilla_model_weights/v_48_030.pt": "c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66",
    "soluble_model_weights/v_48_002.pt": "0877f840978fe770be6fcec025784d8f50c438571db3260c05e41aa207a7c448",
    "ca_model_weights/v_48_002.pt": "ec038b44a987d7c8351b6ed887c82a2370d54e45e55a6bdaf508a729cef0340e",
}

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProteinMPNN-main")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="权重所在根目录")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true", help="校验仓库内权重 SHA-256")
    g.add_argument("--list", action="store_true", help="仅列出期望 SHA-256 与来源")
    args = ap.parse_args()

    if args.list:
        for rel, h in WEIGHTS.items():
            print(f"{h}  {rel}")
        print("\n来源: 官方 dauparas/ProteinMPNN 仓库 model_weights/（第三方权重，非本项目的）")
        return 0

    missing, ok, bad = [], 0, []
    for rel, expected in WEIGHTS.items():
        path = os.path.join(args.root, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        actual = sha256_of(path)
        if actual == expected:
            ok += 1
        else:
            bad.append((rel, actual, expected))

    for rel in missing:
        print(f"[MISSING] {rel}  (请从官方 dauparas/ProteinMPNN 获取 model_weights，或用 --list 看期望哈希)")
    for rel, a, e in bad:
        print(f"[MISMATCH] {rel}\n  实=%s\n  期=%s" % (a, e))
    print(f"[RESULT] 匹配 {ok} / 共 {len(WEIGHTS)}；缺失 {len(missing)}；不一致 {len(bad)}")
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
