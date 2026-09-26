#!/usr/bin/env python3
"""
APEH v2.11 应急还原 — 把 mdp_files/ 恢复为某次运行前的备份

正常路径下 start_simulation.py 跑完会自动还原，本脚本用于：
  * 不确定当前 mdp 是不是默认值时，手动恢复一次；
  * 上一次运行被强制中断、想立刻还原时。

用法:
  python restore_mdp.py              # 列出所有备份
  python restore_mdp.py --restore    # 从最近的备份还原
  python restore_mdp.py --restore --from 20260926_153222
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_all  # noqa: E402

PROJECT_ROOT = run_all.GROMACS_DIR
MDP_DIR = PROJECT_ROOT / "mdp_files"
RUNS_DIR = PROJECT_ROOT / "mdp_runs"
ACTIVE_MARKER = RUNS_DIR / "_active.json"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref_temps(backup_dir: Path) -> str:
    """读出该备份里 nvt/npt/md 的 ref-t，便于判断哪份才是默认值。"""
    parts = []
    for name in ("nvt.mdp", "npt.mdp", "md.mdp"):
        path = backup_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?:^|\n)\s*ref-t\s*=\s*([\d.]+)", text)
        parts.append(f"{name[:-4]}={match.group(1) if match else '?'}")
    return ", ".join(parts) or "（无 mdp）"


def list_backups() -> list[tuple[str, Path]]:
    if not RUNS_DIR.is_dir():
        return []
    found = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        backup = run_dir / "original"
        if backup.is_dir():
            found.append((run_dir.name, backup))
    return found


def restore(backup_dir: Path) -> int:
    if not MDP_DIR.is_dir():
        print(f"[ERROR] 找不到 mdp 目录: {MDP_DIR}")
        return 2
    failures = 0
    for source in sorted(backup_dir.iterdir()):
        if not source.is_file():
            continue
        destination = MDP_DIR / source.name
        shutil.copy2(source, destination)
        print(f"  还原 {source.name}  ({ref_temps(backup_dir)})")
    for name in ("nvt.mdp", "npt.mdp", "md.mdp"):
        original = backup_dir / name
        current = MDP_DIR / name
        if not original.is_file() or not current.is_file():
            continue
        if sha256_of(original) != sha256_of(current):
            print(f"[ERROR] {name} SHA256 校验不一致")
            failures += 1
    if failures == 0:
        ACTIVE_MARKER.unlink(missing_ok=True)
        print(f"[INFO]  还原完成，SHA256 校验通过: {MDP_DIR}")
        return 0
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="restore_mdp.py",
        description="把 mdp_files/ 恢复为某次运行前的备份",
    )
    parser.add_argument("--restore", action="store_true", help="执行还原（默认只列出备份）")
    parser.add_argument("--from", dest="stamp", default="",
                        help="指定备份时间戳，例如 20260926_153222")
    args = parser.parse_args(argv)

    backups = list_backups()
    print(f"mdp 目录: {MDP_DIR}")
    print(f"当前温度: {ref_temps(MDP_DIR)}")
    print()
    if not backups:
        print("没有可用备份（mdp_runs/ 下无 original/ 目录）")
        return 0

    print(f"共 {len(backups)} 个备份:")
    for stamp, backup in backups:
        print(f"  {stamp}   {ref_temps(backup)}")
    print()

    if not args.restore:
        print("加 --restore 执行还原（默认还原最近一次备份）")
        return 0

    if args.stamp:
        matches = [b for s, b in backups if s == args.stamp]
        if not matches:
            print(f"[ERROR] 找不到备份: {args.stamp}")
            return 2
        target = matches[0]
    else:
        target = backups[-1][1]

    print(f"从 {target} 还原 ...")
    return restore(target)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
