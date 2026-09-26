#!/usr/bin/env python3
r"""
APEH v2.11 — 干实验（dry-lab）全流程主控脚本

一条命令串起木聚糖酶改造的干实验流水线：

  Step 0   改名 PDB        input_pdb\*.pdb  -> input_rename\protein.pdb
  Step 1   拓扑/溶剂化/离子 01_pdb2gmx       -> MD_result\pdb2gmx\
  Step 2   能量最小化 EM    02_em            -> MD_result\em\
  Step 3   NVT 平衡         03_nvt           -> MD_result\nvt\
  Step 4   NPT 平衡         04_npt           -> MD_result\npt\
  Step 5   成品模拟 MD      05_md            -> MD_result\md\
  Step 6   清理归档         06_cleanup       -> output_statistics\<批次>\
  Step 7   后处理分析       07_analysis      -> <批次>\analysis\*.csv
  Step 8   候选位置推荐     mutation_position_recommender.py
  Step 9   候选序列设计     ProteinMPNN
  Step 10  活性打分 + TopN  aaindex_model（AAindex 逐位点模型）
  Step 11  结果归集         collect_results.py -> result\<批次>\

工作目录（GROMACS 模板 + 步骤脚本 + ProteinMPNN + 打分模型）由 src_code_dir.txt
指定，默认指向 ..\work\src\code；也可用环境变量 APEH_SRC_CODE_DIR 覆盖。

用法（解释器用 conda 环境 apeh，不要用裸 python）:
  python run_all.py                                        # 默认 300 K 协议
  python run_all.py --dry-run                              # 只打印命令，不执行
  python run_all.py --no-gpu --gmx gmx
  python run_all.py --existing-stats-dir output_statistics\20260926_120000
  python run_all.py --design-positions "A35 A68 A102"
  python start_simulation.py --celsius 90                  # 指定温度跑全流程
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 强制 Python 及所有子进程使用 UTF-8，避免 Windows GBK 解码崩溃
os.environ["PYTHONUTF8"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =========================================================================
#  路径解析
# =========================================================================
# 目录约定:
#   v2.11\
#     code\                 本脚本 + aaindex_model\ + collect_results.py
#     result\               Step 11 结果副本: result\<批次>\<子脚本>\
#     src_code_dir.txt      一行, 写明工作目录 src\code 的绝对路径
#     work\src\code\        工作目录本体: GROMACS\ 、ProteinMPNN\ 、fitness_model\
#     work\results\         ProteinMPNN 候选与打分结果
#
# 定位优先级: 环境变量 -> src_code_dir.txt -> ../work/src/code -> 向上搜索

HERE = Path(__file__).resolve().parent           # v2.11\code
PROJECT_ROOT = HERE.parent                       # v2.11
AAINDEX_MODEL_DIR = HERE / "aaindex_model"       # 打分模型包
AAINDEX_MODEL_NAME = "aaindex_nested_cv_auto6"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result"    # Step 11 归集根目录
REPORT_NAME = "run_all_report.txt"               # 总览报告（collect_results 已收录）
_SRC_CODE_HINT = "src_code_dir.txt"


def _read_src_code_hint(script_dir: Path) -> Optional[Path]:
    """读取 src_code_dir.txt 首行的 src/code 绝对路径（允许 # 注释行）。"""
    for base in (script_dir, script_dir.parent, script_dir.parent.parent):
        hint = base / _SRC_CODE_HINT
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
            candidate = Path(line.strip().strip('"'))
            if (candidate / "GROMACS").is_dir():
                return candidate
            break
    return None


def _default_work_src(script_dir: Path) -> Path:
    r"""v2.11 自带工作目录: v2.11\work\src\code。"""
    return script_dir.parent / "work" / "src" / "code"


def _find_src_code_dir(script_dir: Path) -> Path:
    """定位复用的工作目录（以 GROMACS/ 存在为判据）。"""
    env = os.environ.get("APEH_SRC_CODE_DIR", "").strip().strip('"')
    if env:
        candidate = Path(env)
        if (candidate / "GROMACS").is_dir():
            return candidate

    hinted = _read_src_code_hint(script_dir)
    if hinted is not None:
        return hinted

    default = _default_work_src(script_dir)
    if (default / "GROMACS").is_dir():
        return default

    for base in (script_dir, *script_dir.parents):
        candidates = [base / "src" / "code"]
        candidates += sorted(base.glob("*/src/code"))
        for candidate in candidates:
            if (candidate / "GROMACS").is_dir():
                return candidate
    return default


_SCRIPT_CODE_DIR = HERE
_SRC_CODE_DIR = _find_src_code_dir(_SCRIPT_CODE_DIR)
GROMACS_DIR = _SRC_CODE_DIR / "GROMACS"


# =========================================================================
#  默认配置与输出目录
# =========================================================================

DEFAULTS = {
    "gmx_cmd": "gmx",
    "gpu": True,
    "box_type": "dodecahedron",
    "box_dist": "1.0",
    "salt_conc": "0.15",
}

RESULT_PDB2GMX = "MD_result/pdb2gmx"
RESULT_EM = "MD_result/em"
RESULT_NVT = "MD_result/nvt"
RESULT_NPT = "MD_result/npt"
RESULT_MD = "MD_result/md"
TOPOL_DIR = RESULT_PDB2GMX          # NVT/NPT/MD 读取拓扑文件的目录


# =========================================================================
#  日志
# =========================================================================

class Color:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"{Color.GREEN}[INFO]  {_ts()}  {msg}{Color.RESET}")


def warn(msg: str) -> None:
    print(f"{Color.YELLOW}[WARN]  {_ts()}  {msg}{Color.RESET}")


def error(msg: str) -> None:
    print(f"{Color.RED}[ERROR] {_ts()}  {msg}{Color.RESET}", file=sys.stderr)


def banner(text: str) -> None:
    print(f"\n{Color.CYAN}{Color.BOLD}{'=' * 70}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}  {text}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 70}{Color.RESET}\n")


def step_header(num: int, name: str) -> None:
    print(f"\n{Color.MAGENTA}{Color.BOLD}{'-' * 70}{Color.RESET}")
    print(f"{Color.MAGENTA}{Color.BOLD}  >>> Step {num}: {name} <<<{Color.RESET}")
    print(f"{Color.MAGENTA}{Color.BOLD}{'-' * 70}{Color.RESET}\n")


# run_gmx 延迟到 _setup_paths() 把 GROMACS 与 scripts/ 加入 sys.path 后再导入
run_gmx = None


def _setup_paths() -> Path:
    """切换 cwd 到 GROMACS 工作目录，并把 GROMACS/ 与 scripts/ 加入 sys.path。"""
    project_root = GROMACS_DIR
    scripts_dir = project_root / "scripts"

    if not scripts_dir.is_dir():
        sys.exit(
            f"错误: 工作目录不完整，找不到 scripts/: {scripts_dir}\n"
            f"      当前解析到的工作目录: {_SRC_CODE_DIR}\n"
            f"      请确认 {_SRC_CODE_HINT} 首行指向含 GROMACS/ 的 src\\code 目录，\n"
            f"      或设置环境变量 APEH_SRC_CODE_DIR。"
        )

    os.chdir(project_root)
    info(f"已切换工作目录: {project_root}")

    for path in (project_root, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    global run_gmx
    from scripts._gmx_utils import run_gmx as _run_gmx  # noqa: E402
    run_gmx = _run_gmx

    init_file = scripts_dir / "__init__.py"
    if not init_file.is_file():
        init_file.write_text("# GROMACS MD 步骤脚本包\n", encoding="utf-8")

    return project_root


# =========================================================================
#  前置条件检查
# =========================================================================

_STEP_PREREQUISITES: dict[str, dict[str, str]] = {
    "Step 1: 准备步骤": {
        "input_rename/protein.pdb": "改名后的 PDB 结构文件",
    },
    "Step 2: EM": {
        "MD_result/pdb2gmx/topol.top": "拓扑文件",
        "MD_result/pdb2gmx/*_solv_ions.gro": "溶剂化+离子化的结构文件",
    },
    "Step 3: NVT": {
        "MD_result/em/em.gro": "能量最小化后的结构文件",
    },
    "Step 4: NPT": {
        "MD_result/nvt/nvt.gro": "NVT 平衡后的结构文件",
        "MD_result/nvt/nvt.cpt": "NVT 平衡的检查点文件",
    },
    "Step 5: MD": {
        "MD_result/npt/npt.gro": "NPT 平衡后的结构文件",
        "MD_result/npt/npt.cpt": "NPT 平衡的检查点文件",
    },
}


def ensure_prerequisites(step_label: str) -> None:
    """检查上一步骤的输出文件是否存在，缺失则提前停止。"""
    checks = _STEP_PREREQUISITES.get(step_label)
    if checks is None:
        return

    info(f"前置检查 - {step_label}")
    missing = [
        f"  - {description}: 未找到 {pattern}"
        for pattern, description in checks.items()
        if not list(Path().glob(pattern))
    ]
    if missing:
        raise RuntimeError(
            f"上一步骤的输出文件缺失，无法继续执行 {step_label}:\n"
            + "\n".join(missing)
            + "\n请确认上一步骤已成功完成并生成了预期的输出文件。"
        )
    info(f"  前置检查通过 ({len(checks)} 项)")


def run_step(results: dict, key: str, label: Optional[str], func, *args, **kwargs):
    """执行步骤并把结果登记到 results (供总览报告使用)。

    即使步骤抛异常 (含前置检查失败) 也会先登记失败条目再上抛,
    否则报告会漏记失败步骤, 误显示"所有已执行步骤均成功"。
    """
    try:
        if label:
            ensure_prerequisites(label)
        result = func(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - 中断/异常都要留下记录
        results[key] = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise
    results[key] = result if isinstance(result, dict) else {"success": True}
    return result


# =========================================================================
#  Step 0: PDB 改名
# =========================================================================

def step_rename_pdb(project_root: Path) -> Path:
    """调用 renamepdb.py，把 input_pdb/ 的 PDB 复制改名为 input_rename/protein.pdb。"""
    step_header(0, "renamepdb - PDB 文件改名")

    rename_script = project_root / "renamepdb.py"
    pdb_dir = project_root / "input_pdb"
    target_pdb = project_root / "input_rename" / "protein.pdb"

    if not rename_script.is_file():
        raise RuntimeError(f"找不到 renamepdb.py: {rename_script}")
    if not pdb_dir.is_dir():
        raise RuntimeError(f"input_pdb/ 目录不存在: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(
            f"input_pdb/ 下没有 .pdb 文件: {pdb_dir}\n"
            f"请放入待改造蛋白的 PDB，或用 --pdb <文件> 指定。"
        )
    info(f"检测到 {len(pdb_files)} 个 PDB 文件: {[f.name for f in pdb_files]}")

    info("运行 renamepdb.py ...")
    try:
        result = run_gmx(
            [sys.executable, str(rename_script)],
            cwd=str(project_root),
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("renamepdb.py 执行超时 (60s)")
    except FileNotFoundError:
        raise RuntimeError(f"找不到 Python 解释器: {sys.executable}")

    if result.returncode != 0:
        detail = (result.stderr or "").strip()[:500] or "(无输出)"
        raise RuntimeError(
            f"renamepdb.py 返回非零退出码 {result.returncode}:\n{detail}"
        )
    for line in (result.stdout or "").strip().splitlines():
        info(f"  {line.strip()}")

    if not target_pdb.is_file():
        raise RuntimeError(f"改名后未找到目标文件: {target_pdb}")
    info(f"PDB 文件就绪: {target_pdb}")
    return target_pdb


# =========================================================================
#  Step 1~5: MD 主链
# =========================================================================

def step_preparation(pdb_path: str, gmx_cmd: str, box_type: str, box_dist: str,
                     salt_conc: str, dry_run: bool) -> dict:
    """Step 1: 01_pdb2gmx.run_preparation() - 拓扑 + 盒子 + 溶剂化 + 离子中和。"""
    step_header(1, "01_pdb2gmx - 拓扑生成 + 溶剂化 + 离子中和")
    info(f"输入 PDB: {pdb_path}")
    info(f"盒子类型: {box_type}, 边缘距离: {box_dist} nm, 盐浓度: {salt_conc} mol/L")

    prep = importlib.import_module("01_pdb2gmx")
    result = prep.run_preparation(
        pdb_path=pdb_path,
        result_dir=RESULT_PDB2GMX,
        gmx_cmd=gmx_cmd,
        box_type=box_type,
        box_dist=box_dist,
        salt_conc=salt_conc,
        dry_run=dry_run,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"准备步骤失败:\n"
            f"  力场: {result.get('forcefield_name', '?')}\n"
            f"  水模型: {result.get('water_name', '?')}\n"
            f"  错误: {result.get('error', '未知')}\n"
            f"  报告: {result.get('report_path', 'N/A')}"
        )
    info(f"准备步骤完成 - 力场: {result['forcefield_name']}, "
         f"水模型: {result['water_name']}, 盒子: {result['box_type']}")
    info(f"遍历报告: {result['report_path']}")
    return result


def step_em(gmx_cmd: str, gpu: bool, dry_run: bool) -> dict:
    """Step 2: 02_em.run_em() - 能量最小化。"""
    step_header(2, "02_em - 能量最小化 (EM)")
    em = importlib.import_module("02_em")
    result = em.run_em(
        input_dir=RESULT_PDB2GMX,
        output_dir=RESULT_EM,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"能量最小化失败:\n"
            f"  错误: {result.get('error', '未知')}\n"
            f"  报告: {result.get('report_path', 'N/A')}"
        )
    info(f"EM 完成 - Fmax: {result.get('fmax_final', '?')}, "
         f"Epot: {result.get('epot_initial', '?')} -> {result.get('epot_final', '?')}")
    info(f"遍历报告: {result['report_path']}")
    return result


def step_nvt(gmx_cmd: str, gpu: bool, dry_run: bool,
             topol_dir: str = TOPOL_DIR) -> dict:
    """Step 3: 03_nvt.run_nvt() - NVT 温度平衡。"""
    step_header(3, "03_nvt - NVT 温度平衡")
    nvt = importlib.import_module("03_nvt")
    result = nvt.run_nvt(
        input_dir=RESULT_EM,
        output_dir=RESULT_NVT,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
        topol_dir=topol_dir,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"NVT 温度平衡失败:\n"
            f"  错误: {result.get('error', '未知')}\n"
            f"  报告: {result.get('report_path', 'N/A')}"
        )
    info(f"NVT 完成 - 温度: {result.get('temp_average', '?')} K "
         f"(目标: {result.get('temp_target', '?')} K)")
    info(f"遍历报告: {result['report_path']}")
    return result


def step_npt(gmx_cmd: str, gpu: bool, dry_run: bool,
             topol_dir: str = TOPOL_DIR) -> dict:
    """Step 4: 04_npt.run_npt() - NPT 压力平衡。"""
    step_header(4, "04_npt - NPT 压力平衡")
    npt = importlib.import_module("04_npt")
    result = npt.run_npt(
        input_dir=RESULT_NVT,
        output_dir=RESULT_NPT,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
        topol_dir=topol_dir,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"NPT 压力平衡失败:\n"
            f"  错误: {result.get('error', '未知')}\n"
            f"  报告: {result.get('report_path', 'N/A')}"
        )
    info(f"NPT 完成 - 温度: {result.get('temp_average', '?')} K, "
         f"压力: {result.get('pres_average', '?')} bar, "
         f"密度: {result.get('dens_average', '?')} kg/m3")
    info(f"遍历报告: {result['report_path']}")
    return result


def step_md(gmx_cmd: str, gpu: bool, dry_run: bool,
            topol_dir: str = TOPOL_DIR) -> dict:
    """Step 5: 05_md.run_md() - 成品模拟。"""
    step_header(5, "05_md - 成品模拟 (Production MD)")
    md = importlib.import_module("05_md")
    result = md.run_md(
        input_dir=RESULT_NPT,
        output_dir=RESULT_MD,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
        topol_dir=topol_dir,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"成品模拟失败:\n"
            f"  错误: {result.get('error', '未知')}\n"
            f"  报告: {result.get('report_path', 'N/A')}"
        )
    sim_ns = (result.get("sim_time", 0) or 0) / 1000.0
    info(f"MD 完成 - 温度: {result.get('temp_average', '?')} K, "
         f"压力: {result.get('pres_average', '?')} bar, "
         f"密度: {result.get('dens_average', '?')} kg/m3, 时长: {sim_ns:.1f} ns")
    info(f"轨迹文件: {result.get('traj_file', '?')}")
    info(f"遍历报告: {result['report_path']}")
    return result


# =========================================================================
#  Step 6/7: 清理归档与后处理分析（失败不阻塞主流程）
# =========================================================================

def step_cleanup(project_root: Path, dry_run: bool) -> dict:
    """Step 6: 06_cleanup.run_cleanup() - 清理临时文件并归档 MD 核心产物。"""
    step_header(6, "06_cleanup - 清理临时文件 + 数据归档")
    cleanup = importlib.import_module("06_cleanup")
    result = cleanup.run_cleanup(project_root=str(project_root), dry_run=dry_run)

    if result.get("success"):
        info(f"核心产物已归档: {result['stats_dir']}")
        info(f"临时文件已归档: {result['temp_dir']}")
    else:
        warn(f"清理过程有 {len(result.get('errors', []))} 个非致命错误")
    for line in result.get("summary", []):
        info(f"  {line}")
    return result


def step_analysis(project_root: Path, gmx_cmd: str, dry_run: bool,
                  stats_dir: Optional[str] = None) -> dict:
    """Step 7: 07_analysis.run_analysis() - RMSF / SASA / 能量分解等分析。"""
    step_header(7, "07_analysis - 后处理分析 (结构量 / 表面-溶剂 / 能量分解)")
    analysis = importlib.import_module("07_analysis")
    result = analysis.run_analysis(
        project_root=str(project_root),
        stats_dir=stats_dir,
        gmx_cmd=gmx_cmd,
        dry_run=dry_run,
    )
    if result.get("success"):
        info(f"分析完成 - 产物目录: {result['analysis_dir']}")
    else:
        warn(f"分析部分完成, 有 {len(result.get('errors', []))} 个非致命错误")
    for line in result.get("summary", []):
        info(f"  {line}")
    return result


# =========================================================================
#  Step 8/9 共用工具: 末态结构提取与单运行目录装配
# =========================================================================

def _add_single_chain_id(source_pdb: Path, output_pdb: Path, chain_id: str) -> None:
    """为 GROMACS 导出的单链 PDB 补回链标识。"""
    if not re.fullmatch(r"[A-Za-z0-9]", chain_id):
        raise RuntimeError("ProteinMPNN 链名必须是一个字母或数字：" + chain_id)

    atom_count = 0
    output_lines: list[str] = []
    for line in source_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            if len(line) < 22:
                raise RuntimeError(f"PDB 原子记录格式异常：{line}")
            line = line[:21] + chain_id + line[22:]
            atom_count += 1
        output_lines.append(line)

    if atom_count == 0:
        raise RuntimeError(f"导出的 PDB 中没有原子记录：{source_pdb}")
    output_pdb.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _extract_final_protein_pdb(stats_path: Path, gmx_cmd: str,
                               design_chain: str) -> Path:
    """从 MD 末态坐标提取单链蛋白 PDB 并补回链名。"""
    stats_path = stats_path.resolve()
    tpr_path = stats_path / "md.tpr"
    gro_path = stats_path / "md.gro"
    for required_path in (tpr_path, gro_path):
        if not required_path.is_file():
            raise RuntimeError(f"末态结构提取缺少文件：{required_path}")

    analysis_dir = stats_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb = analysis_dir / "final_protein_raw.pdb"
    final_pdb = analysis_dir / f"final_protein_chain{design_chain}.pdb"

    info("正在从 MD 最终坐标提取蛋白结构……")
    trjconv = subprocess.run(
        [gmx_cmd, "trjconv", "-s", str(tpr_path), "-f", str(gro_path),
         "-o", str(raw_pdb), "-center", "-pbc", "mol"],
        input="Protein\nProtein\n",
        text=True,
        capture_output=True,
        cwd=str(analysis_dir),
        check=False,
    )
    if trjconv.returncode != 0 or not raw_pdb.is_file():
        details = (trjconv.stderr or trjconv.stdout or "没有错误输出")[-3000:]
        raise RuntimeError(
            f"提取 MD 最终蛋白 PDB 失败。\n退出码：{trjconv.returncode}\n{details}"
        )

    _add_single_chain_id(raw_pdb, final_pdb, design_chain)
    raw_pdb.unlink(missing_ok=True)
    info(f"末态蛋白 PDB：{final_pdb}")
    return final_pdb


def _resolve_mpnn_run_dir(project_root: Path, stats_path: Path,
                          output_dir: Optional[str]) -> Path:
    """计算 ProteinMPNN 单运行目录（位置推荐与序列设计共用）。"""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    repository_root = project_root.parents[2]      # …\work
    return repository_root / "results" / "gromacs_proteinmpnn" / stats_path.name


def _assemble_run_dir(run_dir: Path, final_pdb: Path, analysis_dir: Path) -> None:
    """把末态 PDB 与 GROMACS 分析文件装配到单运行目录。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_pdb, run_dir / final_pdb.name)
    for name in ("rmsf.csv", "residue_sasa.csv"):
        source = analysis_dir / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)


# -------------------------------------------------------------------------
#  ProteinMPNN 目录解析
# -------------------------------------------------------------------------

_MPNN_MODEL_NAME = "v_48_020"
_MPNN_SCRIPT = "protein_mpnn_run.py"
_MPNN_HELPERS = (
    "parse_multiple_chains.py",
    "assign_fixed_chains.py",
    "make_fixed_positions_dict.py",
)


def _validate_proteinmpnn_dir(candidate: Path) -> Optional[str]:
    """校验 ProteinMPNN 目录是否完整可运行；返回问题描述或 None。"""
    if not (candidate / _MPNN_SCRIPT).is_file():
        return f"缺少主程序：{candidate / _MPNN_SCRIPT}"
    weight = candidate / "vanilla_model_weights" / f"{_MPNN_MODEL_NAME}.pt"
    if not weight.is_file():
        return (
            f"缺少模型权重：{weight}。请从 ProteinMPNN 官方发行版下载 "
            f"{_MPNN_MODEL_NAME}.pt 放入 vanilla_model_weights/ 目录。"
        )
    for helper in _MPNN_HELPERS:
        helper_path = candidate / "helper_scripts" / helper
        if not helper_path.is_file():
            return f"缺少辅助脚本：{helper_path}"
    return None


def _resolve_proteinmpnn_dir(cli_dir: str) -> tuple[Optional[Path], Optional[str]]:
    """解析 ProteinMPNN 目录，返回 (路径或 None, 问题描述或 None)。"""
    if cli_dir.strip():
        candidate = Path(cli_dir).expanduser().resolve()
        if not candidate.is_dir():
            return None, f"路径不存在：{candidate}"
        reason = _validate_proteinmpnn_dir(candidate)
        return (None, reason) if reason else (candidate, None)

    default = _SRC_CODE_DIR / "ProteinMPNN" / "ProteinMPNN-main"
    if default.is_dir():
        reason = _validate_proteinmpnn_dir(default)
        return (None, reason) if reason else (default, None)
    return None, None


# =========================================================================
#  Step 8: 候选突变位置推荐
# =========================================================================

def step_recommend_positions(project_root: Path, gmx_cmd: str, stats_dir: str,
                             design_chain: str, top_k: int,
                             exclude_positions: str, min_separation: int) -> dict:
    """Step 8: 依据逐残基 RMSF 与 SASA 生成候选突变位置建议。"""
    step_header(8, "GROMACS 数据 -> 候选突变位置建议")

    stats_path = Path(stats_dir).resolve()
    analysis_dir = stats_path / "analysis"
    final_pdb = _extract_final_protein_pdb(stats_path, gmx_cmd, design_chain)

    rmsf_csv = analysis_dir / "rmsf.csv"
    residue_sasa_csv = analysis_dir / "residue_sasa.csv"
    for required_path in (rmsf_csv, residue_sasa_csv):
        if not required_path.is_file():
            raise RuntimeError(
                f"位置推荐缺少分析文件：{required_path}；"
                "请先用 07_analysis.py 重新执行 Step 7"
            )

    repository_root = project_root.parents[2]
    recommender = (repository_root / "src" / "code" / "ProteinMPNN"
                   / "mutation_position_recommender.py")
    if not recommender.is_file():
        raise RuntimeError(f"找不到位置推荐脚本：{recommender}")

    run_dir = _resolve_mpnn_run_dir(project_root, stats_path, None)
    _assemble_run_dir(run_dir, final_pdb, analysis_dir)

    command = [
        sys.executable, str(recommender),
        "--dir", str(run_dir),
        "--chain", design_chain,
        "--top-k", str(top_k),
        "--min-separation", str(min_separation),
    ]
    if exclude_positions.strip():
        command.extend(["--exclude-positions", exclude_positions.strip()])

    completed = subprocess.run(command, check=False)
    recommended_csv = run_dir / "recommended_positions.csv"
    if completed.returncode != 0 or not recommended_csv.is_file():
        raise RuntimeError(
            "候选突变位置推荐失败。"
            f"退出码：{completed.returncode}；预期结果：{recommended_csv}"
        )

    info(f"推荐位置表：{recommended_csv}")
    warn("推荐仅基于 RMSF 和 SASA，必须人工排除催化、结合及保守位点")
    return {
        "success": True,
        "input_pdb": str(final_pdb),
        "recommended_csv": str(recommended_csv),
        "all_scores_csv": str(run_dir / "position_scores.csv"),
        "run_dir": str(run_dir),
    }


# =========================================================================
#  Step 9: ProteinMPNN 候选序列设计
# =========================================================================

def step_proteinmpnn(project_root: Path, gmx_cmd: str, stats_dir: str,
                     proteinmpnn_dir: str, design_chain: str, design_positions: str,
                     num_sequences: int, sampling_temp: float, seed: int,
                     omit_aas: str = "", output_dir: Optional[str] = None,
                     from_recommendations: bool = False) -> dict:
    """Step 9: 提取 MD 末态蛋白结构并调用 ProteinMPNN 生成候选序列。"""
    step_header(9, "人工确认位置 -> ProteinMPNN")

    stats_path = Path(stats_dir).resolve()
    analysis_dir = stats_path / "analysis"
    final_pdb = _extract_final_protein_pdb(stats_path, gmx_cmd, design_chain)
    info(f"ProteinMPNN 输入 PDB：{final_pdb}")

    repository_root = project_root.parents[2]
    wrapper_path = (repository_root / "src" / "code" / "ProteinMPNN"
                    / "proteinmpnn_wrapper.py")
    if not wrapper_path.is_file():
        raise RuntimeError(f"找不到 ProteinMPNN 封装脚本：{wrapper_path}")

    run_dir = _resolve_mpnn_run_dir(project_root, stats_path, output_dir)
    _assemble_run_dir(run_dir, final_pdb, analysis_dir)

    recommended_csv = run_dir / "recommended_positions.csv"
    if from_recommendations and not recommended_csv.is_file():
        raise RuntimeError(
            f"自动使用位置推荐时找不到推荐表：{recommended_csv}；"
            "请先确保 Step 8 位置推荐成功运行"
        )

    command = [
        sys.executable, str(wrapper_path),
        "--dir", str(run_dir),
        "--chains", design_chain,
        "--num-sequences", str(num_sequences),
        "--sampling-temp", str(sampling_temp),
        "--seed", str(seed),
    ]
    if design_positions.strip():
        command.extend(["--design-positions", design_positions.strip()])
    elif from_recommendations:
        command.append("--from-recommendations")
    if proteinmpnn_dir:
        command.extend(["--proteinmpnn-dir",
                        str(Path(proteinmpnn_dir).expanduser().resolve())])
    if omit_aas:
        command.extend(["--omit-aas", omit_aas])

    info("正在运行 ProteinMPNN……")
    completed = subprocess.run(command, check=False)
    candidates_csv = run_dir / "candidates.csv"
    audited_csv = run_dir / "audited_candidates.csv"
    if completed.returncode != 0 or not candidates_csv.is_file():
        raise RuntimeError(
            f"ProteinMPNN 运行失败。退出码：{completed.returncode}；"
            f"预期结果：{candidates_csv}"
        )
    if not audited_csv.is_file():
        raise RuntimeError(f"没有找到候选审核表：{audited_csv}")

    info(f"ProteinMPNN 候选结果：{candidates_csv}")
    info(f"ProteinMPNN 审核结果：{audited_csv}")
    return {
        "success": True,
        "input_pdb": str(final_pdb),
        "output_dir": str(run_dir),
        "candidates_csv": str(candidates_csv),
        "audited_candidates_csv": str(audited_csv),
        "design_positions": design_positions,
        "from_recommendations": from_recommendations,
    }


# =========================================================================
#  Step 10: aaindex_model 活性打分 -> TopN 推荐
# =========================================================================

def step_scoring(run_dir: str, design_chain: str, artifacts_dir: str = "",
                 top_k: int = 20, min_pred: float = 0.0,
                 focus_lengths: str = "4,5,6,7", priority_positions: str = "",
                 per_value_cap: int = 5,
                 similarity_threshold: float = 0.8) -> dict:
    """Step 10: 对 Step 9 的候选做 AAindex 活性打分并输出 TopN 推荐。"""
    step_header(10, "aaindex_model - MPNN 候选打分 -> TopN 推荐")

    run_dir_path = Path(run_dir).resolve()
    audited = run_dir_path / "audited_candidates.csv"
    if not audited.is_file():
        raise RuntimeError(f"找不到候选审核表：{audited}；请先运行 ProteinMPNN (Step 9)")

    run_loop = AAINDEX_MODEL_DIR / "run_loop.py"
    if not run_loop.is_file():
        raise RuntimeError(f"找不到 AAindex 活性闭环脚本：{run_loop}")

    if not artifacts_dir:
        artifacts_dir = str(AAINDEX_MODEL_DIR / "artifacts")
    artifacts_path = Path(artifacts_dir).expanduser().resolve()
    if not (artifacts_path / "final_pipeline.pkl").is_file():
        raise RuntimeError(
            f"找不到 AAindex 活性模型产物：{artifacts_path / 'final_pipeline.pkl'}"
        )

    command = [
        sys.executable, str(run_loop),
        "--dir", str(run_dir_path),
        "--artifacts", str(artifacts_path),
        "--chain", design_chain,
        "--top-k", str(top_k),
        "--min-pred", str(min_pred),
        "--focus-lengths", focus_lengths,
        "--per-value-cap", str(per_value_cap),
        "--similarity-threshold", str(similarity_threshold),
    ]
    if priority_positions.strip():
        command += ["--priority-positions", priority_positions.strip()]

    info("正在运行 AAindex 活性闭环（打分 + TopN 选择）……")
    completed = subprocess.run(command, check=False)
    scored = run_dir_path / "scored_candidates.csv"
    top = run_dir_path / "topN_recommendations.csv"
    if completed.returncode != 0 or not top.is_file():
        raise RuntimeError(
            f"AAindex 活性闭环运行失败。退出码：{completed.returncode}；预期结果：{top}"
        )

    # TopN 为空（只有表头）不得判为成功 —— 防止空结果被计入成功
    try:
        with open(top, encoding="utf-8", errors="replace") as handle:
            top_rows = [
                row for row in csv.DictReader(handle)
                if any((value or "").strip() for value in row.values())
            ]
    except Exception:  # noqa: BLE001
        top_rows = []
    if not top_rows:
        raise RuntimeError(
            f"AAindex 活性闭环输出为空：{top} 含 0 条推荐。"
            "TopN 为空表明所有候选均未通过蛋白特异校验或低于阈值，应视为失败而非成功"
            "（请核对输入 PDB 与训练蛋白是否同源、阈值是否合理）。"
        )

    if scored.is_file():
        try:
            with open(scored, encoding="utf-8", errors="replace") as handle:
                rows = list(csv.DictReader(handle))
            n_valid = sum(1 for row in rows
                          if str(row.get("score_invalid", "1")).strip() == "0")
            if rows and n_valid == 0:
                warn("所有候选均未通过蛋白特异校验 (score_invalid=1)，TopN 为空。")
                warn("请确认 MPNN 输入 PDB 与 AAindex 模型训练数据为同一蛋白。")
        except Exception as exc:  # noqa: BLE001
            warn(f"统计有效候选失败（不影响结果）: {exc}")

    info(f"打分表：{scored}")
    info(f"TopN 推荐：{top}")
    return {
        "success": True,
        "input_dir": str(run_dir_path),
        "scored_csv": str(scored),
        "topn_csv": str(top),
        "scorer": AAINDEX_MODEL_NAME,
    }


# =========================================================================
#  环境检测
# =========================================================================

def check_gromacs(gmx_cmd: str) -> str:
    """检测 GROMACS 是否可用，返回版本信息。"""
    info(f"检测 GROMACS 环境: {gmx_cmd}")
    try:
        result = run_gmx([gmx_cmd, "--version"], capture_output=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError(
            f"找不到 GROMACS 命令 '{gmx_cmd}'。\n"
            f"请确认 GROMACS 已安装并加入 PATH（或 conda 环境已激活），\n"
            f"也可以用 --gmx <完整路径> 指定。"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"GROMACS 命令 '{gmx_cmd}' 执行超时")

    if result.returncode != 0:
        raise RuntimeError(f"GROMACS 命令 '{gmx_cmd}' 返回非零退出码")
    version_line = (result.stdout or "").strip().split("\n")[0]
    info(f"GROMACS 版本: {version_line}")
    return version_line


def check_project_structure(project_root: Path) -> None:
    """检查工作目录结构是否完整。"""
    required = {
        "输入 PDB 目录": project_root / "input_pdb",
        "MDP 参数目录": project_root / "mdp_files",
        "脚本目录": project_root / "scripts",
    }
    missing = [name for name, path in required.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "工作目录结构不完整, 缺少:\n"
            + "\n".join(f"  - {name}: {required[name]}" for name in missing)
        )

    mdp_dir = project_root / "mdp_files"
    missing_mdp = [
        name for name in ("em.mdp", "nvt.mdp", "npt.mdp", "md.mdp", "ions.mdp")
        if not (mdp_dir / name).is_file()
    ]
    if missing_mdp:
        warn(f"缺少 MDP 参数文件: {missing_mdp}")
    info("工作目录结构检查通过")


# =========================================================================
#  物理合理性守卫与总览报告
# =========================================================================

def _empty_analysis_csvs(analysis_dir: Optional[str]) -> list[str]:
    """返回分析目录中数据行数为 0 的 CSV 文件名。"""
    if not analysis_dir:
        return []
    directory = Path(analysis_dir)
    if not directory.is_dir():
        return []
    empty = []
    for csv_path in sorted(directory.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                data_rows = sum(
                    1 for row in reader if any(cell.strip() for cell in row)
                )
            if data_rows == 0:
                empty.append(csv_path.name)
        except Exception:  # noqa: BLE001
            empty.append(csv_path.name)
    return empty


def _step_physical_problems(key: str, result: dict) -> list[str]:
    """物理合理性守卫：温度 / 压力 / 时长 / 分析行数必须落在可信区间。"""
    problems: list[str] = []
    if key in ("nvt", "npt", "md"):
        try:
            temp = float(result.get("temp_average"))
            if temp != temp or not (150 <= temp <= 400):
                problems.append(f"平均温度不可信: {result.get('temp_average')} K")
        except (TypeError, ValueError):
            problems.append(f"平均温度缺失/非数值: {result.get('temp_average')} K")
    if key in ("npt", "md"):
        try:
            pressure = float(result.get("pres_average"))
            if pressure != pressure:
                problems.append(f"压力非数值: {result.get('pres_average')} bar")
        except (TypeError, ValueError):
            problems.append(f"压力缺失: {result.get('pres_average')} bar")
    if key == "md":
        sim_ns = (result.get("sim_time", 0) or 0) / 1000.0
        if sim_ns <= 0:
            problems.append(f"模拟时长非正: {sim_ns} ns")
    if key == "analysis":
        problems.extend(_empty_analysis_csvs(result.get("analysis_dir")))
    return problems


# 报告中显示的步骤顺序与标题
_REPORT_STEPS: tuple[tuple[str, str], ...] = (
    ("rename_pdb", "Step 0  PDB 改名"),
    ("preparation", "Step 1  拓扑/溶剂化/离子中和"),
    ("em", "Step 2  能量最小化"),
    ("nvt", "Step 3  NVT 平衡"),
    ("npt", "Step 4  NPT 平衡"),
    ("md", "Step 5  成品模拟 MD"),
    ("cleanup", "Step 6  清理归档"),
    ("analysis", "Step 7  后处理分析"),
    ("recommendation", "Step 8  候选位置推荐"),
    ("proteinmpnn", "Step 9  ProteinMPNN 序列设计"),
    ("scoring", "Step 10 AAindex 活性打分 + TopN"),
    ("collect", "Step 11 结果归集"),
)

# 每步在报告中额外展示的字段
_REPORT_METRICS: dict[str, tuple[str, ...]] = {
    "preparation": ("forcefield_name", "water_name", "box_type"),
    "em": ("fmax_final", "epot_initial", "epot_final"),
    "nvt": ("temp_average", "temp_target"),
    "npt": ("temp_average", "pres_average", "dens_average"),
    "md": ("temp_average", "pres_average", "dens_average", "sim_time"),
    "cleanup": ("stats_dir", "temp_dir"),
    "analysis": ("analysis_dir",),
    "recommendation": ("recommended_csv",),
    "proteinmpnn": ("output_dir", "candidates_csv", "audited_candidates_csv"),
    "scoring": ("topn_csv", "scored_csv", "scorer"),
}


def write_summary_report(project_root: Path, start_time: datetime, results: dict,
                         gromacs_version: str, gmx_cmd: str, gpu: bool,
                         completion: str) -> Path:
    """生成全流程总览报告 MD_result/run_all_report.txt。"""
    report_path = project_root / "MD_result" / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    end_time = datetime.now()

    lines = [
        "=" * 78,
        "APEH v2.11 干实验全流程总览报告",
        "=" * 78,
        f"开始时间 : {start_time:%Y-%m-%d %H:%M:%S}",
        f"结束时间 : {end_time:%Y-%m-%d %H:%M:%S}",
        f"总耗时   : {end_time - start_time}",
        f"结束状态 : {completion}",
        f"工作目录 : {project_root}",
        f"GROMACS  : {gromacs_version}  (命令: {gmx_cmd})",
        f"GPU 加速 : {'是' if gpu else '否 (仅 CPU)'}",
        "",
        "-" * 78,
        "步骤结果",
        "-" * 78,
    ]

    for key, title in _REPORT_STEPS:
        result = results.get(key)
        if result is None:
            continue
        status = "成功" if result.get("success") else "失败"
        if result.get("skipped"):
            status = "跳过"
        lines.append(f"[{status}] {title}")
        for field in _REPORT_METRICS.get(key, ()):
            if result.get(field) is not None:
                lines.append(f"         {field}: {result[field]}")
        if not result.get("success"):
            lines.append(f"         错误: {result.get('error', '未知')}")
        lines.append("")

    failing = [
        title for key, title in _REPORT_STEPS
        if results.get(key) and not results[key].get("success")
    ]
    lines.append("-" * 78)
    if failing:
        lines.append("未全部成功，失败步骤：")
        lines.extend(f"  - {title}" for title in failing)
    else:
        lines.append("所有已执行步骤均成功。")
        if completion not in ("成功", "已有结果处理完成"):
            lines.append(f"注意：流程未正常结束（结束状态: {completion}），请检查上方步骤是否完整。")
    lines.append("=" * 78)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# =========================================================================
#  Step 11: 结果归集
# =========================================================================

def step_collect(stats_dir: Optional[str], mpnn_run_dir: Optional[str],
                 result_dir: str, dry_run: bool) -> dict:
    """Step 11: 调用 collect_results.py，把本次产物复制到 result/<批次>/。"""
    step_header(11, "结果归集 - 按子脚本备份到 result/")

    if dry_run:
        info("dry-run 模式: 跳过结果归集")
        return {"success": True, "skipped": True}

    script = _SCRIPT_CODE_DIR / "collect_results.py"
    if not script.is_file():
        warn(f"找不到归集脚本: {script}")
        return {"success": False, "error": "collect_results.py 不存在"}

    if not stats_dir and not mpnn_run_dir:
        info("本次没有可归集的产物目录，跳过")
        return {"success": True, "skipped": True}

    command = [sys.executable, str(script), "--quiet"]
    if result_dir:
        command += ["--out", str(result_dir)]
    if mpnn_run_dir:
        command += ["--run-dir", str(mpnn_run_dir)]
    if stats_dir:
        command += ["--stats-dir", str(stats_dir)]

    info(f"归集来源: MD 归档={stats_dir or '无'} | 候选打分={mpnn_run_dir or '无'}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        warn("结果归集未完成（不影响主流程；可稍后单独运行 collect_results.py）")
        return {"success": False, "error": f"退出码 {completed.returncode}"}
    return {"success": True}


# =========================================================================
#  命令行
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="APEH v2.11 干实验全流程主控 (GROMACS -> ProteinMPNN -> AAindex TopN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all.py                                        # 默认 300 K 协议
  python run_all.py --dry-run                              # 只预览，不执行
  python run_all.py --no-gpu                               # 仅使用 CPU
  python run_all.py --box-type cubic --salt-conc 0.2
  python run_all.py --design-positions "A35 A68 A102"       # 指定设计位点
  python run_all.py --existing-stats-dir output_statistics\\20260926_120000
  python run_all.py --no-collect                           # 不自动归集结果
""",
    )
    parser.add_argument("--gmx", default=DEFAULTS["gmx_cmd"],
                        help=f"GROMACS 命令 (默认: {DEFAULTS['gmx_cmd']})")
    parser.add_argument("--gpu", action="store_true", default=DEFAULTS["gpu"],
                        dest="gpu", help="使用 GPU 加速 (默认)")
    parser.add_argument("--no-gpu", action="store_false", default=DEFAULTS["gpu"],
                        dest="gpu", help="仅使用 CPU")
    parser.add_argument("--pdb", default="",
                        help="显式指定输入 PDB（复制到工作目录 input_pdb/，非破坏性）")
    parser.add_argument("--box-type", default=DEFAULTS["box_type"],
                        choices=["cubic", "dodecahedron", "triclinic", "octahedron"],
                        help=f"盒子类型 (默认: {DEFAULTS['box_type']})")
    parser.add_argument("--box-dist", default=DEFAULTS["box_dist"],
                        help=f"盒子边缘到蛋白最短距离, nm (默认: {DEFAULTS['box_dist']})")
    parser.add_argument("--salt-conc", default=DEFAULTS["salt_conc"],
                        help=f"盐浓度, mol/L (默认: {DEFAULTS['salt_conc']})")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="预览模式 - 只打印命令不执行")
    parser.add_argument("--proteinmpnn-dir", default="",
                        help="ProteinMPNN 仓库目录；默认用工作目录内 ProteinMPNN-main")
    parser.add_argument("--design-positions", default="",
                        help='覆盖推荐位置，直接指定设计位点，例如 "A35 A68 A102"')
    parser.add_argument("--design-chain", default="A",
                        help="单链蛋白恢复后的链名，默认 A")
    parser.add_argument("--num-sequences", type=int, default=20,
                        help="ProteinMPNN 候选数量，默认 20")
    parser.add_argument("--sampling-temp", type=float, default=0.1,
                        help="ProteinMPNN 采样温度，默认 0.1")
    parser.add_argument("--omit-aas", default="",
                        help='禁止 ProteinMPNN 引入的氨基酸，例如 "PC"')
    parser.add_argument("--seed", type=int, default=37,
                        help="ProteinMPNN 随机种子，默认 37")
    parser.add_argument("--proteinmpnn-output", default="",
                        help="ProteinMPNN 输出目录；默认 results/gromacs_proteinmpnn/<批次>")
    parser.add_argument("--existing-stats-dir", default="",
                        help="复用已有 output_statistics/<批次>，跳过 MD 模拟")
    parser.add_argument("--recommend-positions", action="store_true",
                        help="根据逐残基 RMSF 和 SASA 生成候选突变位置建议")
    parser.add_argument("--recommend-top-k", type=int, default=10,
                        help="推荐候选位置数量，默认 10")
    parser.add_argument("--exclude-positions", default="",
                        help='人工排除的功能或保守位点，例如 "A50 A120"')
    parser.add_argument("--recommend-min-separation", type=int, default=3,
                        help="推荐位置之间的最小链内间隔，默认 3")

    scoring_group = parser.add_mutually_exclusive_group()
    scoring_group.add_argument("--scoring", dest="scoring", action="store_true",
                               default=True,
                               help="ProteinMPNN 后自动做 AAindex 打分并输出 TopN (默认启用)")
    scoring_group.add_argument("--no-scoring", dest="scoring", action="store_false",
                               help="禁用 AAindex 打分闭环 (Step 10)")
    parser.add_argument("--scoring-artifacts", default="",
                        help="AAindex 模型产物目录；默认 code/aaindex_model/artifacts")
    parser.add_argument("--scoring-top-k", type=int, default=20,
                        help="TopN 推荐数量，默认 20")
    parser.add_argument("--scoring-min-pred", type=float, default=0.0,
                        help="AAindex 活性预测值下限，默认 0.0")
    parser.add_argument("--scoring-focus-lengths", default="4,5,6,7",
                        help="聚焦的突变数量范围，默认 4,5,6,7")
    parser.add_argument("--scoring-priority-positions", default="",
                        help="优先考虑的序列位置")
    parser.add_argument("--scoring-per-value-cap", type=int, default=5,
                        help="每个突变数量 value 的推荐上限，默认 5")
    parser.add_argument("--scoring-similarity-threshold", type=float, default=0.8,
                        help="Jaccard 相似度上限，默认 0.8")

    collect_group = parser.add_mutually_exclusive_group()
    collect_group.add_argument("--collect", dest="collect", action="store_true",
                               default=True,
                               help="结束后按子脚本归集产物到 result/<批次> (默认启用)")
    collect_group.add_argument("--no-collect", dest="collect", action="store_false",
                               help="不自动归集结果")
    parser.add_argument("--result-dir", default="",
                        help=f"结果归集根目录；默认 {DEFAULT_RESULT_ROOT}")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> tuple[bool, str]:
    """参数校验，返回 (是否自动串联 ProteinMPNN, ProteinMPNN 目录解析问题)。"""
    resolved_mpnn_dir, mpnn_reason = _resolve_proteinmpnn_dir(args.proteinmpnn_dir)
    if args.proteinmpnn_dir.strip() and not resolved_mpnn_dir:
        parser.error(f"--proteinmpnn-dir 无效：{mpnn_reason}")

    auto_mpnn = resolved_mpnn_dir is not None
    if args.design_positions.strip() and not auto_mpnn:
        parser.error("填写 --design-positions 时必须同时填写一个有效的 --proteinmpnn-dir")
    if args.existing_stats_dir and not (
        auto_mpnn or args.recommend_positions or args.design_positions
    ):
        parser.error(
            "使用 --existing-stats-dir 时，至少还需填写 "
            "--proteinmpnn-dir、--recommend-positions 或 --design-positions"
        )
    if args.design_positions and not re.fullmatch(
        r"[A-Za-z0-9]\d+(?:\s+[A-Za-z0-9]\d+)*", args.design_positions.strip()
    ):
        parser.error('--design-positions 格式应类似 "A35 A68 A102"')
    if not re.fullmatch(r"[A-Za-z0-9]", args.design_chain):
        parser.error("--design-chain 必须是一个字母或数字")
    if args.design_positions:
        wrong_chain = [token for token in args.design_positions.split()
                       if token[0] != args.design_chain]
        if wrong_chain:
            parser.error("设计位置的链名必须与 --design-chain 一致：" + " ".join(wrong_chain))
    if args.num_sequences <= 0:
        parser.error("--num-sequences 必须大于 0")
    if args.sampling_temp <= 0:
        parser.error("--sampling-temp 必须大于 0")

    args.omit_aas = args.omit_aas.strip().upper()
    invalid_omit = sorted(set(args.omit_aas) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid_omit:
        parser.error("--omit-aas 包含无效氨基酸字母：" + " ".join(invalid_omit))

    if args.recommend_top_k <= 0:
        parser.error("--recommend-top-k 必须大于 0")
    if args.recommend_min_separation < 0:
        parser.error("--recommend-min-separation 不能小于 0")
    if args.scoring_top_k <= 0:
        parser.error("--scoring-top-k 必须大于 0")
    if args.scoring_per_value_cap <= 0:
        parser.error("--scoring-per-value-cap 必须大于 0")
    if not (0.0 <= args.scoring_similarity_threshold <= 1.0):
        parser.error("--scoring-similarity-threshold 应在 0.0 到 1.0 之间")
    if args.scoring_min_pred < 0:
        parser.error("--scoring-min-pred 不能小于 0")

    return auto_mpnn, mpnn_reason


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    gmx_cmd = args.gmx
    gpu = args.gpu

    # 相对路径必须相对"启动目录"解析 —— _setup_paths() 之后 cwd 会切到工作目录。
    for attr in ("proteinmpnn_output", "pdb"):
        value = getattr(args, attr)
        if value:
            requested = Path(value).expanduser()
            if not requested.is_absolute():
                requested = (Path.cwd() / requested).resolve()
            setattr(args, attr, str(requested))

    auto_mpnn, mpnn_reason = _validate_args(parser, args)
    resolved_mpnn_dir = _resolve_proteinmpnn_dir(args.proteinmpnn_dir)[0]
    need_recommendation = args.recommend_positions or (
        auto_mpnn and not args.design_positions.strip()
    )

    banner("APEH v2.11 干实验全流程")
    info(f"工作目录:     {_SRC_CODE_DIR}")
    info(f"GROMACS 命令: {gmx_cmd}")
    info(f"GPU 加速:     {'是' if gpu else '否 (仅 CPU)'}")
    if args.dry_run:
        warn(">>> DRY-RUN 预览模式 - 只打印命令, 不实际执行 <<<")
    if args.design_positions:
        info(f"ProteinMPNN:  启用，设计位置 {args.design_positions}")
    elif auto_mpnn:
        info("ProteinMPNN:  启用，自动使用 Step 8 推荐位置 (--from-recommendations)")
    else:
        info("ProteinMPNN:  未启用（工作目录内 ProteinMPNN-main 不完整或未指定）")
    if need_recommendation:
        info(f"位置推荐:     启用，输出前 {args.recommend_top_k} 个候选")
    if args.scoring and (args.design_positions or auto_mpnn):
        info(f"AAindex 打分: 启用 (Top-K={args.scoring_top_k})")
    if not auto_mpnn and mpnn_reason:
        warn(
            "检测到 ProteinMPNN-main 但不够完整，已跳过 Step 8/9/10。\n"
            f"  原因：{mpnn_reason}"
        )

    project_root = _setup_paths()
    start_time = datetime.now()
    results: dict = {}
    gromacs_version: str = "未检测 (环境检测未完成)"

    try:
        # ---- 环境检测 ----
        print()
        info("=" * 50)
        info("环境检测")
        info("=" * 50)
        gromacs_version = check_gromacs(gmx_cmd)
        check_project_structure(project_root)

        # ---- 显式输入 PDB：复制到 input_pdb/（非破坏性）----
        if args.pdb and not args.dry_run:
            source_pdb = Path(args.pdb)
            if not source_pdb.is_file():
                raise RuntimeError(f"--pdb 文件不存在：{source_pdb}")
            pdb_dir = project_root / "input_pdb"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            destination = pdb_dir / source_pdb.name
            if destination.resolve() != source_pdb.resolve():
                info(f"复制输入 PDB：{source_pdb} -> {destination}")
                shutil.copy2(source_pdb, destination)

        # ---- dry-run：只显示步骤计划，不落盘 ----
        if args.dry_run:
            info("dry-run: 跳过实际执行，仅显示步骤计划")
            for key, title in _REPORT_STEPS:
                if key == "collect" and not args.collect:
                    continue
                if key == "recommendation" and not need_recommendation:
                    continue
                if key in ("proteinmpnn", "scoring") and not (
                    args.design_positions or auto_mpnn
                ):
                    continue
                if key == "scoring" and not args.scoring:
                    continue
                info(f"  将执行 {title}")
            info("dry-run 完成（未改动任何文件）")
            return

        # ---- 复用已有 GROMACS 结果：跳过 MD，直接推荐 / ProteinMPNN / 打分 ----
        if args.existing_stats_dir:
            stats_path = Path(args.existing_stats_dir).expanduser()
            if not stats_path.is_absolute():
                stats_path = project_root / stats_path

            rec_run_dir = None
            if need_recommendation:
                analysis_result = step_analysis(
                    project_root=project_root, gmx_cmd=gmx_cmd,
                    dry_run=False, stats_dir=str(stats_path),
                )
                if not analysis_result.get("success"):
                    raise RuntimeError(
                        "重新分析 GROMACS 结果失败，无法生成位置建议："
                        + "; ".join(analysis_result.get("errors", []))
                    )
                recommendation = step_recommend_positions(
                    project_root=project_root, gmx_cmd=gmx_cmd,
                    stats_dir=str(stats_path), design_chain=args.design_chain,
                    top_k=args.recommend_top_k,
                    exclude_positions=args.exclude_positions,
                    min_separation=args.recommend_min_separation,
                )
                results["recommendation"] = recommendation
                rec_run_dir = recommendation.get("run_dir")

            if args.design_positions or auto_mpnn:
                forwarded_out = rec_run_dir or args.proteinmpnn_output or None
                mpnn_result = step_proteinmpnn(
                    project_root=project_root, gmx_cmd=gmx_cmd,
                    stats_dir=str(stats_path),
                    proteinmpnn_dir=str(resolved_mpnn_dir),
                    design_chain=args.design_chain,
                    design_positions=args.design_positions.strip(),
                    num_sequences=args.num_sequences,
                    sampling_temp=args.sampling_temp,
                    seed=args.seed,
                    omit_aas=args.omit_aas,
                    output_dir=forwarded_out,
                    from_recommendations=(
                        auto_mpnn and not args.design_positions.strip()
                    ),
                )
                results["proteinmpnn"] = mpnn_result
                info(f"候选 CSV: {mpnn_result['candidates_csv']}")
                info(f"审核 CSV: {mpnn_result['audited_candidates_csv']}")

                if args.scoring:
                    results["scoring"] = step_scoring(
                        run_dir=mpnn_result["output_dir"],
                        design_chain=args.design_chain,
                        artifacts_dir=args.scoring_artifacts,
                        top_k=args.scoring_top_k,
                        min_pred=args.scoring_min_pred,
                        focus_lengths=args.scoring_focus_lengths,
                        priority_positions=args.scoring_priority_positions,
                        per_value_cap=args.scoring_per_value_cap,
                        similarity_threshold=args.scoring_similarity_threshold,
                    )

            banner("已有 GROMACS 结果处理完成")
            if args.recommend_positions and not args.design_positions and not auto_mpnn:
                warn("请人工确认推荐位置后，再用 --design-positions 运行 ProteinMPNN")
            write_summary_report(project_root, start_time, results,
                                 gromacs_version, gmx_cmd, gpu,
                                 "已有结果处理完成")
            return

        # ---- Step 0: PDB 改名 ----
        pdb_path = run_step(results, "rename_pdb", None, step_rename_pdb, project_root)

        # ---- Step 1~5: MD 主链 ----
        run_step(results, "preparation", "Step 1: 准备步骤", step_preparation,
                 pdb_path=str(pdb_path), gmx_cmd=gmx_cmd,
                 box_type=args.box_type, box_dist=args.box_dist,
                 salt_conc=args.salt_conc, dry_run=args.dry_run)

        run_step(results, "em", "Step 2: EM", step_em, gmx_cmd, gpu, args.dry_run)
        run_step(results, "nvt", "Step 3: NVT", step_nvt, gmx_cmd, gpu, args.dry_run)
        run_step(results, "npt", "Step 4: NPT", step_npt, gmx_cmd, gpu, args.dry_run)
        run_step(results, "md", "Step 5: MD", step_md, gmx_cmd, gpu, args.dry_run)

        # ---- Step 6: 清理归档（非致命）----
        try:
            results["cleanup"] = step_cleanup(project_root, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            warn(f"清理归档步骤出错 (非致命): {exc}")
            results["cleanup"] = {"success": False, "error": str(exc)}

        # ---- Step 7: 后处理分析（非致命）----
        try:
            results["analysis"] = step_analysis(
                project_root=project_root, gmx_cmd=gmx_cmd, dry_run=args.dry_run,
                stats_dir=results.get("cleanup", {}).get("stats_dir"),
            )
        except Exception as exc:  # noqa: BLE001
            warn(f"后处理分析步骤出错 (非致命): {exc}")
            results["analysis"] = {"success": False, "error": str(exc)}

        stats_dir = results.get("cleanup", {}).get("stats_dir")

        # ---- Step 8: 候选位置推荐 ----
        if need_recommendation:
            try:
                if not stats_dir:
                    raise RuntimeError("清理步骤未返回 GROMACS 核心产物目录")
                results["recommendation"] = step_recommend_positions(
                    project_root=project_root, gmx_cmd=gmx_cmd,
                    stats_dir=stats_dir, design_chain=args.design_chain,
                    top_k=args.recommend_top_k,
                    exclude_positions=args.exclude_positions,
                    min_separation=args.recommend_min_separation,
                )
            except Exception as exc:  # noqa: BLE001
                error(f"候选突变位置推荐失败: {exc}")
                results["recommendation"] = {"success": False, "error": str(exc)}

        # ---- Step 9: ProteinMPNN 序列设计 ----
        if args.design_positions or auto_mpnn:
            try:
                if not stats_dir:
                    raise RuntimeError("清理步骤未返回 GROMACS 核心产物目录")
                forwarded_out = (results.get("recommendation", {}).get("run_dir")
                                 or args.proteinmpnn_output or None)
                results["proteinmpnn"] = step_proteinmpnn(
                    project_root=project_root, gmx_cmd=gmx_cmd,
                    stats_dir=stats_dir,
                    proteinmpnn_dir=str(resolved_mpnn_dir),
                    design_chain=args.design_chain,
                    design_positions=args.design_positions.strip(),
                    num_sequences=args.num_sequences,
                    sampling_temp=args.sampling_temp,
                    seed=args.seed,
                    omit_aas=args.omit_aas,
                    output_dir=forwarded_out,
                    from_recommendations=(
                        auto_mpnn and not args.design_positions.strip()
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                error(f"ProteinMPNN 对接失败: {exc}")
                results["proteinmpnn"] = {"success": False, "error": str(exc)}

        # ---- Step 10: AAindex 活性打分 -> TopN ----
        if (args.design_positions or auto_mpnn) and args.scoring:
            try:
                mpnn_out = results.get("proteinmpnn", {}).get("output_dir")
                if not mpnn_out:
                    raise RuntimeError("ProteinMPNN 步骤未返回输出目录，无法运行打分")
                results["scoring"] = step_scoring(
                    run_dir=mpnn_out,
                    design_chain=args.design_chain,
                    artifacts_dir=args.scoring_artifacts,
                    top_k=args.scoring_top_k,
                    min_pred=args.scoring_min_pred,
                    focus_lengths=args.scoring_focus_lengths,
                    priority_positions=args.scoring_priority_positions,
                    per_value_cap=args.scoring_per_value_cap,
                    similarity_threshold=args.scoring_similarity_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                error(f"AAindex 活性闭环失败: {exc}")
                results["scoring"] = {"success": False, "error": str(exc)}

        # ---- 物理合理性守卫：不可信结果不计入成功 ----
        for key in ("nvt", "npt", "md", "analysis"):
            result = results.get(key)
            if result and result.get("success") and _step_physical_problems(key, result):
                result["success"] = False
                result.setdefault("error", "物理结果不可信（见报告），不作为有效证据")

        requested_steps = ["rename_pdb", "preparation", "em", "nvt", "npt", "md",
                           "cleanup", "analysis"]
        if need_recommendation:
            requested_steps.append("recommendation")
        if args.design_positions or auto_mpnn:
            requested_steps.append("proteinmpnn")
            if args.scoring:
                requested_steps.append("scoring")

        pipeline_success = all(
            results.get(key, {}).get("success") for key in requested_steps
        )

        banner("全流程完成 - 所有步骤成功!" if pipeline_success
               else "流程完成 - 部分步骤失败")

        report_path = write_summary_report(
            project_root, start_time, results, gromacs_version, gmx_cmd, gpu,
            "成功" if pipeline_success else "部分失败",
        )
        if not (args.design_positions or auto_mpnn or args.recommend_positions):
            warn("本次仅运行 GROMACS MD (Step 0-7)。如需继续跑 ProteinMPNN 与 "
                 "AAindex 打分：加 --proteinmpnn-dir，并可用 --existing-stats-dir "
                 "复用本次 MD 结果。")
        info(f"总览报告: {report_path}")
        traj_root = Path(stats_dir) if stats_dir else project_root / RESULT_MD
        info(f"轨迹文件: {traj_root / 'md.xtc'}")
        info(f"能量文件: {traj_root / 'md.edr'}")

        # ---- Step 11: 结果归集（非关键）----
        if args.collect:
            try:
                results["collect"] = step_collect(
                    stats_dir=stats_dir,
                    mpnn_run_dir=(results.get("proteinmpnn") or {}).get("output_dir"),
                    result_dir=args.result_dir,
                    dry_run=args.dry_run,
                )
            except Exception as exc:  # noqa: BLE001
                warn(f"结果归集异常（不影响主流程）: {type(exc).__name__}: {exc}")
        print()
        sys.exit(0 if pipeline_success else 1)

    except RuntimeError as exc:
        print()
        banner("全流程中断 - 存在失败步骤")
        error(str(exc))
        try:
            report_path = write_summary_report(
                project_root, start_time, results, gromacs_version,
                gmx_cmd, gpu, "中断 - 存在失败步骤",
            )
            info(f"总览报告 (含失败记录): {report_path}")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)

    except KeyboardInterrupt:
        print()
        warn("用户中断 (Ctrl+C)")
        try:
            report_path = write_summary_report(
                project_root, start_time, results, gromacs_version,
                gmx_cmd, gpu, "用户中断 (Ctrl+C)",
            )
            info(f"总览报告: {report_path}")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(130)

    except Exception as exc:  # noqa: BLE001
        print()
        banner("全流程异常终止")
        error(f"未预期的错误: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
