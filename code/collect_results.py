#!/usr/bin/env python3
r"""APEH v2.11 结果归集 — 把一次运行的产物按「子脚本」分目录备份到 v2.11\result\<批次>\

产物原本散落在工作目录的三处（工作目录见 ..\src_code_dir.txt）：
  <src>\GROMACS\MD_result\<步骤>\                      各 MD 步骤的结构/拓扑产物
  <src>\GROMACS\output_statistics\<批次>\                MD 核心文件 + analysis\ 分析 CSV
  <src>\GROMACS\temporary_files\<批次>\<步骤>\           各步骤日志（06_cleanup 归档）
  <src>\..\results\gromacs_proteinmpnn\<批次>\           候选序列 + AAindex 打分 + TopN
本脚本按**子脚本**逐个备份，一个子脚本一个文件夹：

  D:\apeh\v2.11\result\<批次>\
  ├── 00_renamepdb\     ← input_rename\          改名后的 PDB
  ├── 01_pdb2gmx\       ← MD_result\pdb2gmx\     拓扑 + 溶剂化 + 离子化（+ logs\）
  ├── 02_em\            ← MD_result\em\          能量最小化（+ logs\）
  ├── 03_nvt\           ← MD_result\nvt\         NVT 平衡（+ logs\）
  ├── 04_npt\           ← MD_result\npt\         NPT 平衡（+ logs\）
  ├── 05_md\            ← MD_result\md\ + output_statistics\<批次>\md.*   成品模拟（+ logs\）
  ├── 06_cleanup\       ← temporary_files\<批次>\ 归档清单/总览报告
  ├── 07_analysis\      ← output_statistics\<批次>\analysis\   RMSF / SASA / 能量分解
  ├── 08_recommend\     ← 位置打分与推荐、末态单链 PDB
  ├── 09_proteinmpnn\   ← 候选序列、审计表、残基映射、运行日志
  ├── 10_aaindex\       ← scored_candidates.csv、topN_recommendations.csv、aaindex_model_run.json
  ├── report\           ← run_all_aaindex_report.txt（全流程总览）
  └── 结果清单.md        ← 来源路径 + 文件清单 + SHA256

只复制、不移动：工作目录里原始文件保持原位（--existing-stats-dir 等复用功能依赖它们）。

用法
----
  python collect_results.py                      # 归集最新批次
  python collect_results.py --list               # 只列出可用批次
  python collect_results.py --batch 20260910_210501
  python collect_results.py --run-dir <目录>      # 直接指定候选/打分运行目录
  python collect_results.py --no-md-core         # 不含 md.xtc / md.edr 等大文件
  python collect_results.py --no-logs            # 不含各步骤日志
  python collect_results.py --out D:\APEH2.1\result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent           # ...\v2.11\code
PROJECT_ROOT = HERE.parent                       # ...\v2.11
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result"
SRC_CODE_HINT = "src_code_dir.txt"

# 产物文件名 → 归属的子脚本目录（与 run_all.py 的步骤编号一致）
FILE_STEP_RULES = (
    ("10_aaindex", {"scored_candidates.csv", "topN_recommendations.csv",
                    "aaindex_model_run.json"}),
    ("08_recommend", {"position_scores.csv", "recommended_positions.csv"}),
    ("07_analysis", {"rmsf.csv", "residue_sasa.csv"}),
)
# 06_cleanup 归档的 MD 核心文件（体积大，可用 --no-md-core 跳过）
MD_CORE_FILES = ("md.xtc", "md.edr", "md.tpr", "md.gro", "md.log")
# 产物目录名 → 子脚本目录名
MD_STEP_DIRS = (
    ("01_pdb2gmx", "pdb2gmx"),
    ("02_em", "em"),
    ("03_nvt", "nvt"),
    ("04_npt", "npt"),
)
REPORT_FILES = ("run_all_aaindex_report.txt", "run_all_report.txt")
SHA_LIMIT = 32 * 1024 * 1024                     # 超过此大小不计算 SHA256


def info(message: str = "") -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARN]  {message}", flush=True)


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)


# =========================================================================
#  路径解析（与 run_all.py 同一套优先级：环境变量 → 指针文件 → 向上搜索）
# =========================================================================

def find_src_code_dir() -> Optional[Path]:
    env = os.environ.get("APEH_SRC_CODE_DIR", "").strip().strip('"')
    if env:
        cand = Path(env)
        if (cand / "GROMACS").is_dir():
            return cand
    for base in (HERE, HERE.parent, HERE.parent.parent):
        hint = base / SRC_CODE_HINT
        if not hint.is_file():
            continue
        try:
            lines = hint.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cand = Path(line.strip('"'))
            if (cand / "GROMACS").is_dir():
                return cand
            break
    for base in (HERE, *HERE.parents):
        for cand in [base / "src" / "code", *sorted(base.glob("*/src/code"))]:
            if (cand / "GROMACS").is_dir():
                return cand
    return None


def mpnn_root_for(src_code: Path) -> Path:
    """ProteinMPNN 运行目录根（与 run_all.py 的 _resolve_mpnn_run_dir 一致）。"""
    return (src_code / "GROMACS").parents[2] / "results" / "gromacs_proteinmpnn"


def step_dir_for(name: str) -> str:
    for step, names in FILE_STEP_RULES:
        if name in names:
            return step
    if name.lower().endswith(".pdb"):
        return "08_recommend"
    return "09_proteinmpnn"


# =========================================================================
#  批次发现
# =========================================================================

def _batch_dirs(src_code: Path, mpnn_root: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for root, key in ((src_code / "GROMACS" / "output_statistics", "stats_dir"),
                      (src_code / "GROMACS" / "temporary_files", "temp_dir"),
                      (mpnn_root, "run_dir")):
        if not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_dir():
                found.setdefault(path.name, {"batch": path.name})[key] = path
    return found


def list_batches(src_code: Path, mpnn_root: Path) -> list[dict]:
    """列出可归集的批次（按最近修改时间倒序）。"""
    found = _batch_dirs(src_code, mpnn_root)
    for item in found.values():
        stamps = [p.stat().st_mtime for p in
                  (item.get("stats_dir"), item.get("temp_dir"), item.get("run_dir"))
                  if p is not None]
        item["mtime"] = max(stamps) if stamps else 0.0
    return sorted(found.values(), key=lambda x: x["mtime"], reverse=True)


def _file_count(path: Optional[Path]) -> int:
    if path is None or not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


# =========================================================================
#  归集计划
# =========================================================================

def _iter_files(root: Optional[Path]) -> Iterator[Path]:
    if root is None or not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def _parse_stamp(name: str) -> Optional[datetime]:
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def resolve_temp_dir(gromacs: Path, batch: str) -> Optional[Path]:
    """定位该批次对应的步骤日志目录 temporary_files\\<批次>。

    06_cleanup 用同一个 batch_id 同时创建 output_statistics\\<id> 与
    temporary_files\\<id>；但中断重跑会让两者时间戳错开，因此同名单不存在时
    回退到时间上最接近（6 小时内）且有内容的批次目录。
    """
    root = gromacs / "temporary_files"
    if not root.is_dir():
        return None
    exact = root / batch
    if exact.is_dir() and _file_count(exact):
        return exact
    target = _parse_stamp(batch)
    if target is not None:
        best, best_delta = None, None
        for candidate in root.iterdir():
            if not candidate.is_dir() or not _file_count(candidate):
                continue
            stamp = _parse_stamp(candidate.name)
            if stamp is None:
                continue
            delta = abs((stamp - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best, best_delta = candidate, delta
        if best is not None and best_delta is not None and best_delta <= 6 * 3600:
            return best
    return exact if exact.is_dir() else None


def build_plan(src_code: Path, batch: str, source: dict,
               with_md_core: bool = True, with_logs: bool = True) -> list[tuple[Path, Path]]:
    """按子脚本生成 (目标相对路径, 源文件) 清单。"""
    gromacs = src_code / "GROMACS"
    stats_dir: Optional[Path] = source.get("stats_dir")
    run_dir: Optional[Path] = source.get("run_dir")
    temp_dir: Optional[Path] = source.get("temp_dir")
    if temp_dir is None:
        temp_dir = resolve_temp_dir(gromacs, batch)
        source["temp_dir"] = temp_dir          # 供结果清单记录实际使用的日志来源
    plan: list[tuple[Path, Path]] = []

    def add_dir(prefix: str, root: Optional[Path]) -> None:
        for path in _iter_files(root):
            plan.append((Path(prefix) / path.relative_to(root), path))

    def add_flat(prefix: str, files) -> None:
        for path in files:
            if path.is_file():
                plan.append((Path(prefix) / path.name, path))

    # 00 renamepdb
    add_dir("00_renamepdb", gromacs / "input_rename")

    # 01~04 各 MD 步骤：结构产物 + 该步骤日志
    for step, name in MD_STEP_DIRS:
        add_dir(step, gromacs / "MD_result" / name)
        if with_logs:
            add_dir(f"{step}/logs", temp_dir / name if temp_dir else None)

    # 05_md：MD 结果目录 + 06_cleanup 归档出来的 MD 核心文件 + 日志
    add_dir("05_md", gromacs / "MD_result" / "md")
    if with_logs:
        add_dir("05_md/logs", temp_dir / "md" if temp_dir else None)
    if with_md_core and stats_dir and stats_dir.is_dir():
        add_flat("05_md", [stats_dir / name for name in MD_CORE_FILES])

    # 06_cleanup：归档清单（temp 目录根级文件，例如 run_all_report.txt）
    if with_logs and temp_dir and temp_dir.is_dir():
        add_flat("06_cleanup", [p for p in sorted(temp_dir.glob("*")) if p.is_file()])

    # 07_analysis：分析 CSV（以 output_statistics\<批次>\analysis\ 为准）
    if stats_dir and stats_dir.is_dir():
        add_dir("07_analysis", stats_dir / "analysis")
        add_flat("07_analysis", [stats_dir / name for name in
                                 ("rmsf.csv", "residue_sasa.csv", "position_scores.csv",
                                  "recommended_positions.csv")])

    # 08~10：候选/打分运行目录（08 位置推荐已在上面按文件名归类）
    if run_dir and run_dir.is_dir():
        for path in _iter_files(run_dir):
            plan.append((Path(step_dir_for(path.name)) / path.name, path))

    # report：全流程总览
    add_flat("report", [gromacs / "MD_result" / name for name in REPORT_FILES])
    if temp_dir and temp_dir.is_dir():
        add_flat("report", [temp_dir / name for name in REPORT_FILES])

    # 同名去重（先来的优先：GROMACS 原始目录 > 运行目录副本）
    unique: dict[Path, Path] = {}
    for rel, src in plan:
        unique.setdefault(rel, src)
    return sorted(unique.items(), key=lambda kv: str(kv[0]))


# =========================================================================
#  复制 + 清单
# =========================================================================

def _sha256(path: Path) -> str:
    if path.stat().st_size > SHA_LIMIT:
        return "（文件较大，未计算）"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_temperature(src_code: Path) -> Optional[tuple[str, float, float]]:
    runs_root = src_code / "GROMACS" / "mdp_runs"
    if not runs_root.is_dir():
        return None
    stamps = sorted((p for p in runs_root.glob("*") if p.is_dir()),
                    key=lambda p: p.name, reverse=True)
    for stamp in stamps:
        config = stamp / "run_config.json"
        if not config.is_file():
            continue
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        temp_k = data.get("temperature_K")
        temp_c = data.get("temperature_C")
        if isinstance(temp_k, (int, float)):
            return (stamp.name, float(temp_k),
                    float(temp_c) if isinstance(temp_c, (int, float)) else float(temp_k) - 273.15)
    return None


def _human(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 ** 2:.1f} MB"


def collect(batch: str, source: dict, result_root: Path, src_code: Path,
            with_md_core: bool = True, with_logs: bool = True) -> dict:
    dest = result_root / batch
    plan = build_plan(src_code, batch, source, with_md_core, with_logs)

    records = []
    for rel, src in plan:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        records.append({
            "rel": rel.as_posix(),
            "src": src,
            "size": target.stat().st_size,
            "mtime": datetime.fromtimestamp(target.stat().st_mtime),
            "sha256": _sha256(target),
        })
    if records:
        _write_manifest(dest, batch, records, source, src_code, with_md_core, with_logs)
    return {"dest": dest, "count": len(records), "records": records}


def _write_manifest(dest: Path, batch: str, records: list[dict], source: dict,
                    src_code: Path, with_md_core: bool, with_logs: bool) -> Path:
    lines = [
        f"# 结果清单 — 批次 {batch}",
        "",
        f"- 归集时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 工作目录：`{src_code}`",
    ]
    for label, key in (("候选/打分来源", "run_dir"), ("MD 归档来源", "stats_dir"),
                       ("步骤日志来源", "temp_dir")):
        path = source.get(key)
        lines.append(f"- {label}：`{path}`" if path else f"- {label}：（无）")
    lines += [
        "- 本目录是**副本**，供查看、留档与交付；工作目录里的原始文件保持原位，",
        "  `--existing-stats-dir` 等复用功能依赖原始目录，请勿把这里当工作目录。",
        f"- 内容范围：各步骤产物 + {'含' if with_md_core else '不含'} MD 大文件（md.xtc/edr/tpr/gro/log）"
        f" + {'含' if with_logs else '不含'} 步骤日志。",
    ]
    temperature = _latest_temperature(src_code)
    if temperature:
        stamp, temp_k, temp_c = temperature
        lines.append(f"- 最近一次温度设置：{temp_k:.2f} K（{temp_c:.2f} °C），记录于 `mdp_runs\\{stamp}`")

    lines += ["", "## 文件清单", "",
              "| 子脚本目录 / 文件 | 大小 | 修改时间 | SHA256 |", "|---|---|---|---|"]
    for item in records:
        digest = item["sha256"]
        short = digest if digest.startswith("（") else f"`{digest[:16]}…`"
        lines.append(f"| `{item['rel']}` | {_human(item['size'])} | "
                     f"{item['mtime']:%Y-%m-%d %H:%M} | {short} |")

    lines += ["", "## 子脚本目录约定", "", "| 目录 | 对应子脚本 | 内容 |", "|---|---|---|",
              "| `00_renamepdb/` | renamepdb | 改名后的输入 PDB |",
              "| `01_pdb2gmx/` | `01_pdb2gmx.py` | 拓扑、溶剂化与离子化结构 |",
              "| `02_em/` | `02_em.py` | 能量最小化结构与能量 |",
              "| `03_nvt/` | `03_nvt.py` | NVT 平衡结构与检查点 |",
              "| `04_npt/` | `04_npt.py` | NPT 平衡结构与检查点 |",
              "| `05_md/` | `05_md.py` | 成品模拟结构、轨迹、能量、日志 |",
              "| `06_cleanup/` | `06_cleanup.py` | 归档清单与总览报告 |",
              "| `07_analysis/` | `07_analysis.py` | RMSF / SASA / 位置打分等分析 CSV |",
              "| `08_recommend/` | 位置推荐（run_all.py Step 8） | 候选突变位点与推荐表、末态 PDB |",
              "| `09_proteinmpnn/` | ProteinMPNN（Step 9） | 候选序列、审计表、残基映射、日志 |",
              "| `10_aaindex/` | `aaindex_model`（Step 10） | 打分结果、TopN 推荐、模型留档 |",
              "| `report/` | run_all.py | 全流程总览报告 |",
              ""]
    path = dest / "结果清单.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# =========================================================================
#  命令行
# =========================================================================

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把一次运行的产物按子脚本备份到 v2.11\\result\\<批次>\\",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("示例:\n"
                "  python collect_results.py                 # 归集最新批次\n"
                "  python collect_results.py --list\n"
                "  python collect_results.py --batch 20260910_210501\n"
                "  python collect_results.py --run-dir D:\\path\\mpnn_run_dir --name 试验1\n"),
    )
    parser.add_argument("--batch", default="", help="批次名（默认取最新批次）")
    parser.add_argument("--list", dest="list_only", action="store_true",
                        help="只列出可用批次，不做归集")
    parser.add_argument("--run-dir", default="", help="直接指定候选/打分运行目录")
    parser.add_argument("--stats-dir", default="", help="直接指定 output_statistics\\<批次> 目录")
    parser.add_argument("--name", default="", help="配合 --run-dir/--stats-dir 使用的批次名")
    parser.add_argument("--out", default="", help="归集根目录；默认 <v2.11>\\result")
    parser.add_argument("--no-md-core", dest="md_core", action="store_false", default=True,
                        help="不复制 md.xtc / md.edr / md.tpr / md.gro / md.log")
    parser.add_argument("--no-logs", dest="logs", action="store_false", default=True,
                        help="不复制 temporary_files 里的步骤日志")
    parser.add_argument("--quiet", action="store_true", help="精简输出（供 run_all.py 调用）")
    parser.add_argument("--src-code", default="", help="手工指定工作目录（覆盖指针文件）")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    quiet = args.quiet

    src_code = Path(args.src_code).expanduser() if args.src_code else find_src_code_dir()
    if src_code is None or not (src_code / "GROMACS").is_dir():
        error("定位不到工作目录（src/code）；请检查 src_code_dir.txt 或 --src-code")
        return 2
    env_out = os.environ.get("APEH_RESULT_DIR", "").strip()
    result_root = (Path(args.out).expanduser() if args.out
                   else Path(env_out).expanduser() if env_out else DEFAULT_RESULT_ROOT)

    mpnn_root = mpnn_root_for(src_code)
    batches = list_batches(src_code, mpnn_root)

    if not quiet:
        info(f"工作目录: {src_code}")
        info(f"结果目录: {result_root}")

    if args.list_only:
        if not batches:
            info("（没有可归集的批次）")
            return 0
        info("")
        info(f"{'批次':<22}{'MD产物':>8}{'步骤日志':>10}{'候选目录':>10}   最近修改")
        for item in batches:
            run_dir = item.get("run_dir")
            info(f"{item['batch']:<22}"
                 f"{_file_count(item.get('stats_dir')):>8}"
                 f"{_file_count(item.get('temp_dir')):>10}"
                 f"{_file_count(run_dir):>10}   "
                 f"{datetime.fromtimestamp(item['mtime']):%Y-%m-%d %H:%M}")
        return 0

    if args.run_dir or args.stats_dir:
        batch = args.name or Path(args.run_dir or args.stats_dir).name
        source = {
            "batch": batch,
            "run_dir": Path(args.run_dir).expanduser().resolve() if args.run_dir else None,
            "stats_dir": Path(args.stats_dir).expanduser().resolve() if args.stats_dir else None,
            "temp_dir": None,
        }
    else:
        source = None
        if args.batch:
            source = next((item for item in batches if item["batch"] == args.batch), None)
            if source is None:
                error(f"找不到批次: {args.batch}（用 --list 查看可用批次）")
                return 2
        else:
            source = next((item for item in batches
                           if _file_count(item.get("stats_dir")) or item.get("run_dir")),
                          batches[0] if batches else None)
        if source is None:
            error("没有任何批次可归集；先跑一次 run_all.py，或用 --run-dir 指定目录")
            return 2

    batch = source.get("batch") or args.name or "批次"
    outcome = collect(batch, source, result_root, src_code,
                      with_md_core=args.md_core, with_logs=args.logs)

    if outcome["count"] == 0:
        error(f"批次 {batch} 里没有找到可归集的产物")
        return 2

    dest = outcome["dest"]
    if not quiet:
        groups: dict[str, list[dict]] = {}
        for item in outcome["records"]:
            groups.setdefault(item["rel"].split("/")[0], []).append(item)
        info("")
        for group, items in sorted(groups.items()):
            total = sum(i["size"] for i in items)
            info(f"  {group:<18} {len(items):>3} 个文件  {_human(total):>10}")
    info(f"[OK] 已归集 {outcome['count']} 个文件 → {dest}")
    if not quiet:
        info(f"     清单: {dest / '结果清单.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
