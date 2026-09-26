#!/usr/bin/env python3
"""
PDB 文件自动重命名与复制脚本（非破坏性）

将 input_pdb/ 中选定（默认按文件名排序后的第一个）的 .pdb 文件，
以 protein.pdb 复制到 input_rename/。不会删除或改动 input_pdb/ 里的原始文件
（原先用 os.rename 会把源文件"移走"，导致源文件丢失）。

功能已在 run_md.py 的 check_input() → _auto_prepare_pdb() 中集成，
主流程调用 run_all.py 即可自动完成 PDB 发现和准备。此脚本保留用于:
  - 仅需要准备 PDB 而不运行完整 MD 流程
  - 手动调试 PDB 文件时的独立工具
"""

import os
import shutil
import sys

# 项目根目录（脚本所在文件夹）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdb_dir = os.path.join(BASE_DIR, "input_pdb")
save_dir = os.path.join(BASE_DIR, "input_rename")
target_name = "protein.pdb"
save_target = os.path.join(save_dir, target_name)

if not os.path.isdir(pdb_dir):
    print(f"错误：文件夹不存在 {pdb_dir}")
    sys.exit(1)

# 新建存放改名后文件的文件夹，不存在则创建
os.makedirs(save_dir, exist_ok=True)

# 遍历 input_pdb 内所有文件，只处理 pdb 后缀；按文件名排序保证确定性（评审：sort）。
# 排除早已存在的 protein.pdb，避免重复处理。
pdb_files = sorted(
    f for f in os.listdir(pdb_dir)
    if f.lower().endswith(".pdb") and f != target_name
)

if not pdb_files:
    print("input_pdb 文件夹内未找到任何 .pdb 文件")
    sys.exit(0)
if len(pdb_files) > 1:
    print(f"警告：检测到多个pdb文件：{pdb_files}")
    print("仅会将第一个（按名排序）复制为 protein.pdb，其余保留原名")

# 非破坏性复制源文件，不改动、不删除原始输入（评审：no move）
src_file = os.path.join(pdb_dir, pdb_files[0])
if os.path.exists(save_target):
    os.remove(save_target)
shutil.copy2(src_file, save_target)
print(f"已复制：{pdb_files[0]}  →  {save_target}")
