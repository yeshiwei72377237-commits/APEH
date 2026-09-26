#!/usr/bin/env python3
"""
GROMACS MD 后处理子脚本 (Step 6) — 清理临时文件 + 数据归档

在成品模拟 (Step 5 MD) 成功完成后由 run_all_md.py 自动调用。

功能:
  1. 删除 GROMACS 自动备份文件 (#...#)        — 所有步骤目录
  2. 删除全精度轨迹 (.trr)                     — em, nvt, npt, md
  3. 删除 pdb2gmx 中间产物                     — protein_box.gro, protein_solv.gro,
                                                  ions.tpr, temp.topa*
  4. 删除残留临时 .xvg 文件                    — _temp_*.xvg (如有)
  5. 移动 MD 核心产物 → output_statistics/<批次>/
     - md.xtc (压缩轨迹), md.edr (能量), md.tpr (拓扑),
       md.gro (最终结构), md.log (日志)
  6. 移动日志/报告/参数历史 → temporary_files/<批次>/
     - 保持原始子目录结构 (em/, nvt/, npt/, md/, pdb2gmx/)

特性:
  - 操作幂等: 文件不存在则跳过, 不报错
  - 清理失败不阻塞主流程 (MD 成果已产生)
  - 支持 --dry-run 预览模式

调用方式:
  由 run_all_md.py 导入调用:
    cleanup = importlib.import_module("06_cleanup")
    result = cleanup.run_cleanup(project_root)
  也可独立运行:
    python scripts/06_cleanup.py --project-root .
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# =========================================================================
#  工具函数
# =========================================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _generate_batch_id() -> str:
    """生成批次标识符: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================================
#  主入口
# =========================================================================

def run_cleanup(
    project_root: str | Path,
    batch_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """执行 MD 后处理: 清理临时文件 + 归档核心数据。

    按 6 个 Phase 依次执行:
      Phase 1 — 删除 GROMACS 自动备份 (#...#)
      Phase 2 — 删除全精度轨迹 (.trr)
      Phase 3 — 删除 pdb2gmx 中间产物 + 残留 .xvg
      Phase 4 — 移动 MD 核心产物 → output_statistics/<batch>/
      Phase 5 — 移动日志/报告/参数历史 → temporary_files/<batch>/
      Phase 6 — 汇总

    Args:
        project_root: 项目根目录路径
        batch_id: 批次标识符, 为 None 时自动生成 YYYYMMDD_HHMMSS
        dry_run: True 时只打印操作, 不实际执行

    Returns:
        {"success": bool, "batch_id": str, "stats_dir": str, "temp_dir": str,
         "summary": [str], "errors": [str]}
    """
    root = Path(project_root).resolve()
    md_result = root / "MD_result"
    output_stats = root / "output_statistics"
    temp_files = root / "temporary_files"

    if batch_id is None:
        batch_id = _generate_batch_id()

    stats_dir = output_stats / batch_id
    temp_dir = temp_files / batch_id

    summary: list[str] = []
    errors: list[str] = []

    # =====================================================================
    #  内部辅助
    # =====================================================================

    def _info(msg: str) -> None:
        print(f"[INFO]  {_ts()}  {msg}")

    def _warn(msg: str) -> None:
        print(f"[WARN]  {_ts()}  {msg}")

    def _safe_delete(file_path: Path) -> bool:
        """安全删除单个文件, 不存在则静默跳过。"""
        if not file_path.exists():
            return False
        if not file_path.is_file():
            return False
        try:
            if dry_run:
                _info(f"  [DRY-RUN] 将删除: {file_path}")
                return True
            file_path.unlink()
            return True
        except Exception as exc:
            _warn(f"  删除失败: {file_path} — {exc}")
            errors.append(f"删除 {file_path}: {exc}")
            return False

    def _safe_move_file(src: Path, dst: Path) -> bool:
        """移动单个文件, 自动创建目标父目录。"""
        if not src.is_file():
            return False
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                _info(f"  [DRY-RUN] 将移动: {src} → {dst}")
                return True
            shutil.move(str(src), str(dst))
            return True
        except Exception as exc:
            _warn(f"  移动失败: {src} → {dst} — {exc}")
            errors.append(f"移动 {src.name}: {exc}")
            return False

    def _safe_move_dir(src: Path, dst: Path) -> bool:
        """移动整个目录 (原子 rename, 同文件系统下高效)。"""
        if not src.is_dir():
            return False
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                _info(f"  [DRY-RUN] 将移动目录: {src} → {dst}")
                return True
            # 若目标已存在则用 copytree + rmtree 兜底
            if dst.exists():
                _warn(f"  目标已存在, 覆盖: {dst}")
                shutil.rmtree(str(dst))
            shutil.move(str(src), str(dst))
            return True
        except Exception as exc:
            _warn(f"  移动目录失败: {src} → {dst} — {exc}")
            errors.append(f"移动目录 {src.name}: {exc}")
            return False

    def _delete_by_glob(rel_pattern: str, label: str) -> int:
        """在项目根下匹配 glob 并删除, 返回删除数量。"""
        count = 0
        for f in root.glob(rel_pattern):
            if _safe_delete(f):
                count += 1
        if count > 0:
            _info(f"  {label}: 已删除 {count} 个")
        return count

    # =====================================================================
    #  Phase 1 — 删除 GROMACS 自动备份 #...#
    # =====================================================================
    _info("Phase 1/6 — 删除 GROMACS 自动备份 (#...#)")

    step_dirs = ["em", "nvt", "npt", "md", "pdb2gmx"]
    total_backups = 0
    for step in step_dirs:
        n = _delete_by_glob(f"MD_result/{step}/#*#", f"  {step}")
        total_backups += n
    summary.append(f"GROMACS 备份文件: 已删除 {total_backups} 个")

    # =====================================================================
    #  Phase 2 — 删除全精度轨迹 .trr
    # =====================================================================
    _info("Phase 2/6 — 删除全精度轨迹 (.trr)")

    trr_files = [
        md_result / "em" / "em.trr",
        md_result / "nvt" / "nvt.trr",
        md_result / "npt" / "npt.trr",
        md_result / "md" / "md.trr",
    ]
    trr_deleted = 0
    for f in trr_files:
        if _safe_delete(f):
            _info(f"  已删除: {f.parent.name}/{f.name}")
            trr_deleted += 1
        else:
            _info(f"  跳过 (不存在): {f.parent.name}/{f.name}")
    summary.append(f"全精度轨迹: 已删除 {trr_deleted} 个")

    # =====================================================================
    #  Phase 3 — 删除 pdb2gmx 中间产物 + 残留 .xvg
    # =====================================================================
    _info("Phase 3/6 — 删除中间产物")

    pdb2gmx_dir = md_result / "pdb2gmx"
    intermediates = [
        pdb2gmx_dir / "protein_box.gro",
        pdb2gmx_dir / "protein_solv.gro",
        pdb2gmx_dir / "ions.tpr",
    ]
    del_count = 0
    for f in intermediates:
        if _safe_delete(f):
            _info(f"  已删除: pdb2gmx/{f.name}")
            del_count += 1

    # temp.topa* (GROMACS 残留临时文件)
    n = _delete_by_glob("MD_result/pdb2gmx/temp.topa*", "  temp.topa*")
    del_count += n

    # 残留 _temp_*.xvg (gmx energy 分析过程中产生的临时文件, 正常已被清理)
    n = _delete_by_glob("MD_result/*/_temp_*.xvg", "  残留 _temp_*.xvg")
    del_count += n

    summary.append(f"中间产物: 已删除 {del_count} 个")

    # =====================================================================
    #  Phase 4 — 移动 MD 核心产物 → output_statistics/<batch>/
    # =====================================================================
    _info(f"Phase 4/6 — 移动 MD 核心产物 → output_statistics/{batch_id}/")

    md_dir = md_result / "md"
    md_core_files = ["md.xtc", "md.edr", "md.tpr", "md.gro", "md.log"]

    moved_core = 0
    for filename in md_core_files:
        src = md_dir / filename
        dst = stats_dir / filename
        if _safe_move_file(src, dst):
            _info(f"  已移动: {filename}")
            moved_core += 1
        else:
            _warn(f"  跳过 (不存在): {filename}")

    summary.append(
        f"MD 核心产物: 已移动 {moved_core}/{len(md_core_files)} "
        f"→ output_statistics/{batch_id}/"
    )

    # =====================================================================
    #  Phase 5 — 移动日志/报告/参数历史 → temporary_files/<batch>/
    # =====================================================================
    _info(f"Phase 5/6 — 移动日志/报告/参数历史 → temporary_files/{batch_id}/")

    # 5a. 总览报告 (根目录下)
    overall_report = md_result / "run_all_report.txt"
    if _safe_move_file(overall_report, temp_dir / "run_all_report.txt"):
        _info("  已移动: run_all_report.txt")

    # 5b. 各步骤文件 — 保持子目录结构
    # 格式: { 步骤名: { "files": [...], "dirs": [...] } }
    step_manifests: dict[str, dict[str, list[str]]] = {
        "pdb2gmx": {
            "files": ["mdout.mdp", "traverse_report.txt"],
            "dirs": [],
        },
        "em": {
            "files": [
                "em.log", "em.gro", "em.edr", "em.tpr",
                "mdout.mdp", "traverse_report.txt",
            ],
            "dirs": ["adjusted_mdp"],
        },
        "nvt": {
            "files": [
                "nvt.log", "nvt.gro", "nvt.edr", "nvt.tpr", "nvt.cpt",
                "mdout.mdp", "traverse_report.txt",
            ],
            "dirs": ["adjusted_mdp"],
        },
        "npt": {
            "files": [
                "npt.log", "npt.gro", "npt.edr", "npt.tpr", "npt.cpt",
                "npt_prev.cpt", "mdout.mdp", "traverse_report.txt",
            ],
            "dirs": ["adjusted_mdp"],
        },
        "md": {
            "files": ["md.cpt", "mdout.mdp", "traverse_report.txt"],
            "dirs": ["adjusted_mdp"],
        },
    }

    total_moved = 0
    for step_name, manifest in step_manifests.items():
        step_src = md_result / step_name
        step_dst = temp_dir / step_name

        for filename in manifest["files"]:
            if _safe_move_file(step_src / filename, step_dst / filename):
                total_moved += 1

        for dirname in manifest["dirs"]:
            if _safe_move_dir(step_src / dirname, step_dst / dirname):
                total_moved += 1

    _info(f"  共移动 {total_moved} 项 → temporary_files/{batch_id}/")
    summary.append(f"临时文件: 已归档 {total_moved} 项 → temporary_files/{batch_id}/")

    # =====================================================================
    #  Phase 6 — 汇总
    # =====================================================================
    _info("Phase 6/6 — 汇总")
    print()

    if errors:
        _warn(f"清理完成, {len(errors)} 个非致命错误")
        for e in errors:
            _warn(f"  - {e}")
    else:
        _info("清理完成, 无错误")

    _info(f"核心产物目录:  {stats_dir}")
    _info(f"临时文件目录:  {temp_dir}")
    print()

    return {
        "success": len(errors) == 0,
        "batch_id": batch_id,
        "stats_dir": str(stats_dir),
        "temp_dir": str(temp_dir),
        "summary": summary,
        "errors": errors,
    }


# =========================================================================
#  独立运行入口
# =========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="06_cleanup.py",
        description="GROMACS MD 后处理 — 清理临时文件 + 归档核心数据",
    )
    parser.add_argument(
        "--project-root", default=".",
        help="项目根目录 (默认: 当前工作目录)",
    )
    parser.add_argument(
        "--batch-id", default=None,
        help="批次标识符 (默认: 自动生成 YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式 — 只打印操作, 不实际执行",
    )
    args = parser.parse_args()

    result = run_cleanup(
        project_root=args.project_root,
        batch_id=args.batch_id,
        dry_run=args.dry_run,
    )

    sys.exit(0 if result["success"] else 1)
