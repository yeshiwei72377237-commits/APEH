#!/usr/bin/env python3
"""
GROMACS MD 准备步骤自动化脚本
自动控制力场选择、水模型选择和电荷平衡

通过遍历策略自动选择合适的人为参数:
  - 力场选择    (AMBER → CHARMM → GROMOS → OPLS, 按族内编号顺序)
  - 水模型选择  (仅尝试与当前力场族兼容的水模型)
  - 盒子类型    (dodecahedron → cubic → octahedron → triclinic)
  - 溶剂模板    (spc216.gro → tip3p.gro → tip4p.gro)
  - 离子中和    (genion -neutral, 自动检测 SOL 组编号)

覆盖步骤 (标准 MD 模拟准备流程):
  Step 1: pdb2gmx — 生成拓扑文件   (力场×水模型自动遍历)
  Step 2: editconf — 定义模拟盒子
  Step 3: solvate  — 添加溶剂
  Step 4: grompp   — 准备离子添加参数
  Step 5: genion   — 添加抗衡离子, 中和体系电荷

输出:
  - 拓扑文件及结构: topol.top, *.gro, *.itp, ions.tpr
  - 详细遍历报告:   traverse_report.txt

用法:
  # 作为独立脚本
  python 01_pdb2gmx.py --pdb input_rename/protein.pdb

  # 作为模块被 run_md.py 导入 (文件名以数字开头, 需用 importlib)
  import importlib
  prep = importlib.import_module("01_pdb2gmx")
  result = prep.run_preparation(pdb_path="input_rename/protein.pdb")

注意:
  - 如果将此文件放在子目录中 (如 scripts/), 需要确保该目录有 __init__.py,
    导入时使用 importlib.import_module("scripts.01_pdb2gmx")
  - 项目根目录 (mdp_files/ 文件夹所在) 默认为当前工作目录
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

from scripts._gmx_utils import run_gmx
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==================== 配色 ====================

class Color:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ==================== 日志工具 ====================

def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"{Color.GREEN}[INFO]  {_timestamp()}  {msg}{Color.RESET}")


def warn(msg: str) -> None:
    print(f"{Color.YELLOW}[WARN]  {_timestamp()}  {msg}{Color.RESET}")


def error(msg: str) -> None:
    print(f"{Color.RED}[ERROR] {_timestamp()}  {msg}{Color.RESET}", file=sys.stderr)


def banner(text: str) -> None:
    print(f"\n{Color.CYAN}{Color.BOLD}{'=' * 60}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}  {text}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 60}{Color.RESET}\n")


# ==================== 力场映射表 ====================

# GROMACS 常用力场: 编号 → (short_name, display_label)
FORCEFIELD_MAP: Dict[str, Tuple[str, str]] = {
    "1":  ("amber03",             "AMBER03"),
    "2":  ("amber94",             "AMBER94"),
    "3":  ("amber96",             "AMBER96"),
    "4":  ("amber99",             "AMBER99"),
    "5":  ("amber99sb",           "AMBER99SB"),
    "6":  ("amber99sb-ildn",      "AMBER99SB-ILDN"),
    "7":  ("amber99sb-star-ildn", "AMBER99SB*-ILDN"),
    "8":  ("charmm36",            "CHARMM36 (2024+)"),
    "9":  ("charmm36-jul2022",    "CHARMM36 (Jul2022)"),
    "10": ("charmm27",            "CHARMM27"),
    "11": ("gromos43a1",          "GROMOS 43A1"),
    "12": ("gromos43a2",          "GROMOS 43A2"),
    "13": ("gromos45a3",          "GROMOS 45A3"),
    "14": ("gromos53a5",          "GROMOS 53A5"),
    "15": ("gromos53a6",          "GROMOS 53A6"),
    "16": ("gromos54a7",          "GROMOS 54A7"),
    "17": ("opls-aa",             "OPLS-AA/M"),
    "18": ("opls-aam",            "OPLS-AA/L"),
}

# GROMACS 常用水模型: 编号 → (short_name, display_label)
WATER_MAP: Dict[str, Tuple[str, str]] = {
    "1": ("tip3p",     "TIP3P"),
    "2": ("tip4p",     "TIP4P"),
    "3": ("tip5p",     "TIP5P"),
    "4": ("spc",       "SPC"),
    "5": ("spce",      "SPC/E"),
    "6": ("tip4p2005", "TIP4P/2005"),
}


# ==================== 力场优先级 & 水模型兼容配对 ====================

# 遍历优先级: AMBER → CHARMM → GROMOS → OPLS
# 每族内按编号从小到大, 配对族内推荐的水模型
_FF_PRIORITY_GROUPS: List[Tuple[str, List[str]]] = [
    ("AMBER",  ["1", "2", "3", "4", "5", "6", "7"]),
    ("CHARMM", ["8", "9", "10"]),
    ("GROMOS", ["11", "12", "13", "14", "15", "16"]),
    ("OPLS",   ["17", "18"]),
]

# GROMACS 官方推荐的力场族 ↔ 水模型配对
_FF_WATER_COMPAT: Dict[str, List[str]] = {
    "AMBER":  ["1", "2", "6", "5"],    # TIP3P(标准) > TIP4P > TIP4P/2005 > SPC/E
    "CHARMM": ["1", "2"],              # 修改版 TIP3P(标准) > TIP4P
    "GROMOS": ["4", "5"],              # SPC(标准) > SPC/E
    "OPLS":   ["1", "2", "5"],         # TIP3P > TIP4P > SPC/E
}


# ==================== 遍历常量 ====================

# 盒子类型遍历顺序 (从最优开始)
_BOX_TYPE_TRAVERSE_ORDER: List[str] = [
    "dodecahedron",
    "cubic",
    "octahedron",
    "triclinic",
]

# 溶剂模板依次尝试列表
_SOLVENT_TEMPLATES: List[str] = [
    "spc216.gro",
    "tip3p.gro",
    "tip4p.gro",
]
_SOLVATE_TEMPLATE_RETRY = 3  # 每个模板最多重试次数

# grompp -maxwarn 递增序列
_MAXWARN_SEQUENCE: List[int] = [10, 20, 50, 100]

# 全局最大命令尝试次数
_MAX_TOTAL_ATTEMPTS = 200


# ==================== 默认参数 ====================

DEFAULTS = {
    "box_type": "dodecahedron",
    "box_dist": "1.0",
    "salt_conc": "0.15",
    "result_dir": "MD_result/pdb2gmx",
    "gmx": "gmx",
    "dry_run": False,
}


# ==================== 异常类 ====================

class PrepError(Exception):
    """准备步骤中的可恢复错误, 携带命令的 stderr/stdout。"""

    def __init__(self, message: str, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


# =========================================================================
#  电荷分析工具
# =========================================================================

def _sum_atom_charges_in_itp(itp_path: Path) -> Optional[float]:
    """对单个 .itp 文件中 [ atoms ] 段的电荷列求和。"""
    if not itp_path.is_file():
        return None
    content = itp_path.read_text(encoding="utf-8", errors="replace")

    in_atoms = False
    total = 0.0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "atoms" in stripped.lower():
            in_atoms = True
            continue
        if in_atoms and stripped.startswith("["):
            break
        if in_atoms and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            # GROMACS [ atoms ]: nr type resnr residue atom cgnr charge mass
            # charge 在第 7 列 (0-indexed: 6)
            if len(parts) >= 7:
                try:
                    total += float(parts[6])
                except ValueError:
                    pass
    return total


def _find_itp_for_molecule(
    mol_name: str,
    topol_dir: Path,
) -> Optional[Path]:
    """在拓扑目录中查找给定分子名对应的 .itp 文件。"""
    # 常见命名模式
    candidates = [
        topol_dir / f"{mol_name}.itp",
        topol_dir / f"topol_{mol_name}.itp",
        topol_dir / f"{mol_name}_chain_A.itp",
        topol_dir / f"topol_{mol_name}_chain_A.itp",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # 模糊匹配 (文件名包含 mol_name)
    for itp in sorted(topol_dir.glob("*.itp")):
        name_lower = itp.name.lower()
        mol_lower = mol_name.lower().replace("_", "")
        if mol_lower in name_lower.replace("_", ""):
            return itp
    return None


def _known_molecule_charge(mol_name: str) -> Optional[float]:
    """已知分子的标准净电荷。"""
    upper = mol_name.upper().strip()
    # 水分子
    if upper in ("SOL", "WAT", "HOH", "TIP3", "TIP4", "TIP5", "SPC", "SPCE"):
        return 0.0
    # 阳离子
    if upper in ("NA", "NA+", "SOD"):
        return 1.0
    if upper in ("K", "K+", "POT"):
        return 1.0
    if upper in ("CA", "CA2+", "CAL"):
        return 2.0
    if upper in ("MG", "MG2+", "MG"):
        return 2.0
    # 阴离子
    if upper in ("CL", "CL-", "CLA"):
        return -1.0
    return None


def calculate_system_charge(topol_path: Path) -> Optional[float]:
    """解析 topol.top, 计算体系净电荷。

    策略:
      1. 解析 [ molecules ] 段获取 {分子名: 数量}
      2. 对已知分子 (水/离子) 使用内置净电荷
      3. 对蛋白等大分子, 在其 .itp 文件中累加原子电荷
      4. 任何一步失败则返回 None

    Returns:
        体系净电荷 (e), 解析失败返回 None
    """
    if not topol_path.is_file():
        return None

    content = topol_path.read_text(encoding="utf-8", errors="replace")
    topol_dir = topol_path.parent

    # ---------- 解析 [ molecules ] 段 ----------
    molecules: List[Tuple[str, int]] = []
    in_molecules = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "molecules" in stripped.lower():
            in_molecules = True
            continue
        if in_molecules and stripped.startswith("["):
            break
        if in_molecules and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    molecules.append((parts[0], int(parts[1])))
                except ValueError:
                    pass

    if not molecules:
        return None

    # ---------- 对每种分子求和 ----------
    total_charge = 0.0
    for mol_name, count in molecules:
        # 先查已知分子
        known = _known_molecule_charge(mol_name)
        if known is not None:
            total_charge += known * count
            continue

        # 查找 itp 文件并累加原子电荷
        itp_path = _find_itp_for_molecule(mol_name, topol_dir)
        if itp_path is None:
            return None  # 无法解析

        atom_sum = _sum_atom_charges_in_itp(itp_path)
        if atom_sum is None:
            return None
        total_charge += atom_sum * count

    return total_charge


# =========================================================================
#  genion 输出解析
# =========================================================================

def _parse_genion_group_list(stdout: str) -> List[str]:
    """从 genion 的 stdout 中解析所有可用组的编号列表。

    GROMACS 输出格式示例:
        Group     0 (         System) has ...
        Group     1 (        Protein) has ...
        Group    13 (            SOL) has ...
        Select a group:

    返回: ['0', '1', ..., '13', ...]
    """
    pattern = re.compile(r'^\s*Group\s+(\d+)\s*\(', re.MULTILINE | re.IGNORECASE)
    return pattern.findall(stdout)


def _find_solvent_group(stdout: str) -> Optional[str]:
    """从 genion 输出中识别溶剂组 (SOL/WAT) 的编号。

    匹配模式:
        Group    13 (            SOL)
        Group     5 (    Water    )
        Group    20 (         SOL)
    """
    # 精确匹配 SOL / Water / solvent
    pattern = re.compile(
        r'Group\s+(\d+)\s*\(\s*(SOL|WAT|Water|solvent)\s*\)',
        re.IGNORECASE,
    )
    m = pattern.search(stdout)
    if m:
        return m.group(1)

    # 兜底: 取最后一个非 System 组 (通常是 SOL)
    all_groups = _parse_genion_group_list(stdout)
    if len(all_groups) >= 2:
        # 跳过 Group 0 (System), 取最后一个
        non_system = [g for g in all_groups if g != "0"]
        if non_system:
            return non_system[-1]
    return None


def _parse_ion_counts_from_topology(topol_path: Path) -> Tuple[int, int]:
    """从更新后的 topol.top 的 [ molecules ] 段统计 Na⁺/Cl⁻ 数量。

    Returns:
        (na_count, cl_count)
    """
    if not topol_path.is_file():
        return 0, 0

    content = topol_path.read_text(encoding="utf-8", errors="replace")
    na_count = 0
    cl_count = 0

    in_molecules = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "molecules" in stripped.lower():
            in_molecules = True
            continue
        if in_molecules and stripped.startswith("["):
            break
        if in_molecules and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 2:
                name_upper = parts[0].upper()
                try:
                    cnt = int(parts[1])
                except ValueError:
                    continue
                if name_upper in ("NA", "NA+", "SOD"):
                    na_count = cnt
                elif name_upper in ("CL", "CL-", "CLA"):
                    cl_count = cnt

    return na_count, cl_count


# =========================================================================
#  遍历报告
# =========================================================================

class TraverseReport:
    """遍历报告生成器, 记录每一步的详细信息并生成格式化报告。

    报告内容包括:
      - 每次力场/水模型尝试的结果与错误原因
      - 每个后续步骤的执行情况
      - 电荷计算与离子中和详情
      - 最终采用的参数
      - 生成的文件清单
    """

    def __init__(self, pdb_path: str, result_dir: Path) -> None:
        self.pdb_path = pdb_path
        self.result_dir = result_dir
        self.start_time = datetime.now()
        self.entries: List[dict] = []
        self.final_params: dict = {}
        self.generated_files: List[str] = []
        self.success = False

    # ---------- 记录 ----------

    def add_attempt(self, **kwargs) -> None:
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        }
        self.entries.append(entry)

    # ---------- 生成报告文本 ----------

    def generate(self) -> str:
        lines: List[str] = []
        sep = "=" * 80

        # ---- 头部 ----
        lines.append(sep)
        lines.append("GROMACS MD 准备步骤 — 自动遍历报告")
        lines.append(sep)
        lines.append(f"  生成时间:   {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  完成时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  输入 PDB:   {self.pdb_path}")
        lines.append(f"  输出目录:   {self.result_dir}")
        lines.append(f"  总共尝试:   {len(self.entries)} 次")
        lines.append(f"  最终结果:   {'[成功]' if self.success else '[失败]'}")
        lines.append("")

        # ---- 遍历配置 ----
        total_ff = sum(len(nums) for _, nums in _FF_PRIORITY_GROUPS)
        lines.append(sep)
        lines.append("遍历配置")
        lines.append(sep)
        lines.append(f"  力场遍历策略:   AMBER → CHARMM → GROMOS → OPLS")
        lines.append(f"  力场总数:       {total_ff} 种")
        lines.append(f"  水模型配对:     仅尝试与力场族兼容的水模型")
        lines.append(f"  盒子类型遍历:   {' → '.join(_BOX_TYPE_TRAVERSE_ORDER)}")
        lines.append(f"  溶剂模板遍历:   {' → '.join(_SOLVENT_TEMPLATES)}")
        lines.append(f"  离子中和策略:   genion -neutral (自动探测 SOL 组)")
        lines.append("")

        # ---- 遍历详情 ----
        lines.append(sep)
        lines.append("遍历详情")
        lines.append(sep)

        pdb2gmx_entries = [e for e in self.entries if e.get("step") == "pdb2gmx"]
        other_entries = [e for e in self.entries if e.get("step") != "pdb2gmx"]

        if pdb2gmx_entries:
            lines.append(f"\n  >>> pdb2gmx 力场×水模型遍历 ({len(pdb2gmx_entries)} 次) <<<\n")
            for i, entry in enumerate(pdb2gmx_entries, 1):
                status = "成功" if entry.get("status") == "success" else "失败"
                icon = "OK" if status == "成功" else "!!"
                lines.append(f"  [{icon}] 尝试 {i}  |  {entry['timestamp']}")
                lines.append(f"       力场:     {entry.get('forcefield_label', '?')}")
                lines.append(f"       力场名:   {entry.get('forcefield_name', '?')}")
                lines.append(f"       水模型:   {entry.get('water_label', '?')}")
                lines.append(f"       水模型名: {entry.get('water_name', '?')}")
                lines.append(f"       结果:     {status}")
                if entry.get("error_summary"):
                    lines.append(f"       错误摘要: {entry['error_summary']}")
                if entry.get("duration") is not None:
                    lines.append(f"       耗时:     {entry['duration']:.1f}s")
                lines.append("")

        if other_entries:
            lines.append(f"  >>> 后续步骤 ({len(other_entries)} 次) <<<\n")
            for i, entry in enumerate(other_entries, 1):
                status = "成功" if entry.get("status") == "success" else "失败"
                icon = "OK" if status == "成功" else "!!"
                lines.append(f"  [{icon}] {entry.get('step', '?')}  |  {entry['timestamp']}")
                if entry.get("detail"):
                    lines.append(f"       详情:     {entry['detail']}")
                lines.append(f"       结果:     {status}")
                if entry.get("error_summary"):
                    lines.append(f"       错误摘要: {entry['error_summary']}")
                if entry.get("duration") is not None:
                    lines.append(f"       耗时:     {entry['duration']:.1f}s")
                lines.append("")

        # ---- 最终参数 ----
        if self.success and self.final_params:
            lines.append(sep)
            lines.append("最终采用参数")
            lines.append(sep)
            for key, value in self.final_params.items():
                lines.append(f"  {key}:  {value}")
            lines.append("")

        # ---- 电荷平衡 ----
        charge_info = self.final_params.get("_charge_info", {})
        if charge_info:
            lines.append(sep)
            lines.append("电荷平衡详情")
            lines.append(sep)
            for key, value in charge_info.items():
                lines.append(f"  {key}:  {value}")
            lines.append("")

        # ---- 文件清单 ----
        if self.generated_files:
            lines.append(sep)
            lines.append("生成文件清单")
            lines.append(sep)
            for f in self.generated_files:
                lines.append(f"  {f}")
            lines.append("")

        lines.append(sep)
        lines.append("报告结束")
        lines.append(sep)

        return "\n".join(lines)

    # ---------- 保存 ----------

    def save(self) -> Path:
        report_path = self.result_dir / "traverse_report.txt"
        report_path.write_text(self.generate(), encoding="utf-8")
        return report_path


# =========================================================================
#  准备步骤运行器
# =========================================================================

class PreparationRunner:
    """GROMACS MD 准备步骤运行器。

    覆盖步骤:
      Step 1: pdb2gmx — 拓扑生成   (自动遍历力场×水模型)
      Step 2: editconf — 定义盒子
      Step 3: solvate  — 添加溶剂
      Step 4: grompp   — 准备离子添加
      Step 5: genion   — 添加抗衡离子, 中和体系电荷

    用法:
        runner = PreparationRunner(pdb_path="input_rename/protein.pdb")
        result = runner.run()
        # result["success"] → True/False
        # result["forcefield_name"] → 最终采用的力场
        # result["report_path"]   → 遍历报告路径
    """

    def __init__(
        self,
        pdb_path: str,
        result_dir: str = "MD_result/pdb2gmx",
        gmx_cmd: str = "gmx",
        box_type: str = "dodecahedron",
        box_dist: str = "1.0",
        salt_conc: str = "0.15",
        dry_run: bool = False,
    ) -> None:
        # ---- 用户参数 ----
        self.pdb_path = pdb_path
        self.result_dir = Path(result_dir)
        self.gmx_cmd = gmx_cmd
        self.box_type = box_type
        self.box_dist = str(box_dist)
        self.salt_conc = str(salt_conc)
        self.dry_run = dry_run

        # ---- 派生路径 ----
        self.pdb_basename = Path(pdb_path).stem
        self.project_root = Path.cwd()

        # ---- 运行时状态 (在遍历过程中被更新) ----
        self.forcefield_num: str = ""
        self.forcefield_name: str = ""
        self.water_num: str = ""
        self.water_name: str = ""
        self.solv_group_num: str = ""

        # ---- 计数器 & 报告 ----
        self._total_attempts: int = 0
        self.report = TraverseReport(pdb_path, self.result_dir)

    # =====================================================================
    #  内部工具方法
    # =====================================================================

    def _count_attempt(self) -> int:
        """递增尝试计数器, 超过全局上限时抛出异常。"""
        self._total_attempts += 1
        if self._total_attempts > _MAX_TOTAL_ATTEMPTS:
            raise PrepError(
                f"已达全局最大命令尝试次数 ({_MAX_TOTAL_ATTEMPTS}), 终止运行"
            )
        return self._total_attempts

    # -- 文件备份/恢复 --

    def _backup_file(self, target: Path) -> Optional[Path]:
        """备份文件为 .bak。"""
        if not target.is_file():
            return None
        backup = target.with_suffix(target.suffix + ".bak")
        if not self.dry_run:
            shutil.copy2(str(target), str(backup))
        return backup

    def _restore_file(self, target: Path) -> None:
        """从 .bak 恢复文件。"""
        backup = target.with_suffix(target.suffix + ".bak")
        if backup.is_file() and not self.dry_run:
            shutil.move(str(backup), str(target))

    def _cleanup_backup(self, target: Path) -> None:
        """删除 .bak 备份。"""
        backup = target.with_suffix(target.suffix + ".bak")
        if backup.is_file() and not self.dry_run:
            backup.unlink()

    # -- GROMACS 命令执行 --

    def run_gmx(
        self,
        *cmd: str,
        cwd: Optional[Path] = None,
        stdin_input: Optional[str] = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess:
        """执行 GROMACS 命令的统一入口。

        Args:
            *cmd: gmx 之后的命令行参数, 如 "pdb2gmx", "-f", "x.pdb"
            cwd: 工作目录
            stdin_input: 通过标准输入传递给命令的文本
            capture: 是否捕获 stdout/stderr

        Returns:
            CompletedProcess (仅在成功时)

        Raises:
            PrepError: 命令执行失败
        """
        full_cmd = [self.gmx_cmd] + list(cmd)

        if self.dry_run:
            cwd_str = str(cwd) if cwd else "."
            input_info = f"  |  stdin: {stdin_input!r}" if stdin_input else ""
            print(
                f"  {Color.YELLOW}[DRY-RUN]{Color.RESET} "
                f"cd {cwd_str} && {' '.join(full_cmd)}{input_info}"
            )
            return subprocess.CompletedProcess(args=full_cmd, returncode=0)

        try:
            result = run_gmx(
                full_cmd,
                cwd=str(cwd) if cwd else None,
                input=stdin_input,
                capture_output=capture,
            )
            result.check_returncode()
            return result
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr or ""
            stdout_text = e.stdout or ""
            error(f"命令失败 (返回码 {e.returncode}): {' '.join(full_cmd)}")
            if stderr_text:
                # 只打印前 500 字符避免刷屏
                short = stderr_text.strip()[:500]
                print(f"{Color.RED}{short}{Color.RESET}", file=sys.stderr)
            raise PrepError(
                f"命令失败: {' '.join(full_cmd)}",
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e
        except FileNotFoundError:
            raise PrepError(
                f"找不到 GROMACS 命令 '{self.gmx_cmd}', "
                f"请确认已安装并激活 GROMACS 环境"
            ) from None

    # -- 错误摘要提取 --

    @staticmethod
    def _extract_error_summary(stderr: str, stdout: str) -> str:
        """从命令输出中提取关键错误信息 (用于报告)。"""
        combined = (stderr + "\n" + stdout).strip()
        if not combined:
            return "(无输出)"

        lines = combined.splitlines()
        # 优先返回含 "Error", "Fatal", "not found" 的行
        keywords = ["fatal", "error", "not found", "unknown", "did not match",
                     "residue", "atom type", "cannot"]
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                return line.strip()[:200]

        # 否则返回最后 3 行
        return " | ".join(line.strip() for line in lines[-3:] if line.strip())[:200]

    # =====================================================================
    #  力场/水模型组合生成
    # =====================================================================

    def _generate_all_ff_water_combos(self) -> List[Tuple[str, str, str]]:
        """生成所有待遍历的 (ff_num, water_num, ff_family) 组合。

        顺序: AMBER → CHARMM → GROMOS → OPLS
        每族内按编号顺序, 配对族推荐的水模型。
        """
        combos: List[Tuple[str, str, str]] = []
        for ff_family, ff_nums in _FF_PRIORITY_GROUPS:
            for ff_num in ff_nums:
                for water_num in _FF_WATER_COMPAT[ff_family]:
                    combos.append((ff_num, water_num, ff_family))
        return combos

    # =====================================================================
    #  Step 1: pdb2gmx — 力场×水模型自动遍历
    # =====================================================================

    def _try_pdb2gmx_once(self) -> subprocess.CompletedProcess:
        """单次执行 pdb2gmx 命令。成功返回 CompletedProcess, 失败抛出 PrepError。"""
        stdin = f"{self.forcefield_num}\n{self.water_num}\n"
        return self.run_gmx(
            "pdb2gmx",
            "-f", str(self.project_root / self.pdb_path),
            "-o", f"{self.pdb_basename}_processed.gro",
            "-p", "topol.top",
            "-i", "posre.itp",
            "-water", self.water_name,
            "-ff", self.forcefield_name,
            "-ignh",
            cwd=self.result_dir,
            stdin_input=stdin,
            capture=True,
        )

    def step_pdb2gmx(self) -> None:
        """Step 1: 遍历力场×水模型组合执行 pdb2gmx, 直到成功。

        遍历策略:
          AMBER 族力场 × AMBER 兼容水模型
          → CHARMM 族力场 × CHARMM 兼容水模型
          → GROMOS 族力场 × GROMOS 兼容水模型
          → OPLS 族力场 × OPLS 兼容水模型
        """
        banner("Step 1/5: pdb2gmx — 生成拓扑文件 (自动遍历力场×水模型)")

        all_combos = self._generate_all_ff_water_combos()
        tried: set = set()

        for ff_num, water_num, ff_family in all_combos:
            if (ff_num, water_num) in tried:
                continue
            tried.add((ff_num, water_num))

            # 更新当前选择
            self.forcefield_num = ff_num
            self.forcefield_name = FORCEFIELD_MAP[ff_num][0]
            self.water_num = water_num
            self.water_name = WATER_MAP[water_num][0]

            attempt = self._count_attempt()
            ff_label = FORCEFIELD_MAP[ff_num][1]
            water_label = WATER_MAP[water_num][1]

            if attempt > 1:
                info(f"--- 第 {attempt} 次尝试 ---")
            info(f"  力场族: {ff_family}")
            info(f"  力场:   [{ff_num}] {ff_label}  ({self.forcefield_name})")
            info(f"  水模型: [{water_num}] {water_label}  ({self.water_name})")

            t0 = time.time()
            try:
                result = self._try_pdb2gmx_once()
                dt = time.time() - t0

                # 打印 GROMACS 输出 (含重要的拓扑生成信息)
                if result.stdout:
                    print(result.stdout)

                info(f"拓扑文件生成完毕: {self.result_dir / 'topol.top'}")

                self.report.add_attempt(
                    step="pdb2gmx",
                    status="success",
                    forcefield_num=ff_num,
                    forcefield_name=self.forcefield_name,
                    forcefield_label=ff_label,
                    water_num=water_num,
                    water_name=self.water_name,
                    water_label=water_label,
                    duration=dt,
                )
                return  # ✅ 成功

            except PrepError as e:
                dt = time.time() - t0
                summary = self._extract_error_summary(e.stderr, e.stdout)

                warn(f"不兼容 — {ff_label} + {water_label}")
                if summary:
                    warn(f"  原因: {summary[:120]}")

                self.report.add_attempt(
                    step="pdb2gmx",
                    status="failed",
                    forcefield_num=ff_num,
                    forcefield_name=self.forcefield_name,
                    forcefield_label=ff_label,
                    water_num=water_num,
                    water_name=self.water_name,
                    water_label=water_label,
                    error_summary=summary,
                    duration=dt,
                )

        raise PrepError(
            f"所有 {len(tried)} 种力场/水模型组合均已尝试完毕, "
            f"均无法为该 PDB 文件生成拓扑。"
            f"\n请检查 PDB 文件是否包含非标准残基或缺少原子。"
        )

    # =====================================================================
    #  Step 2: editconf — 定义模拟盒子
    # =====================================================================

    def step_editconf(self) -> None:
        """Step 2: 定义模拟盒子, 盒子类型不合适时自动降级。

        遍历顺序: dodecahedron → cubic → octahedron → triclinic
        """
        banner("Step 2/5: editconf — 定义模拟盒子")

        traverse_order = [self.box_type] + [
            bt for bt in _BOX_TYPE_TRAVERSE_ORDER if bt != self.box_type
        ]

        for box_type in traverse_order:
            self._count_attempt()
            info(f"  盒子类型: {box_type}, 边缘距离: {self.box_dist} nm")

            t0 = time.time()
            try:
                self.run_gmx(
                    "editconf",
                    "-f", f"{self.pdb_basename}_processed.gro",
                    "-o", f"{self.pdb_basename}_box.gro",
                    "-c",
                    "-d", self.box_dist,
                    "-bt", box_type,
                    cwd=self.result_dir,
                )
                dt = time.time() - t0
                self.box_type = box_type
                info(f"盒子已定义: {self.result_dir / f'{self.pdb_basename}_box.gro'}")

                self.report.add_attempt(
                    step="editconf",
                    status="success",
                    detail=f"盒子类型={box_type}, 距离={self.box_dist} nm",
                    duration=dt,
                )
                return
            except PrepError as e:
                dt = time.time() - t0
                warn(f"盒子类型 '{box_type}' 失败")
                self.report.add_attempt(
                    step="editconf",
                    status="failed",
                    detail=f"盒子类型={box_type}",
                    error_summary=self._extract_error_summary(e.stderr, e.stdout),
                    duration=dt,
                )

        raise PrepError("editconf: 所有盒子类型均已尝试完毕, 仍未通过")

    # =====================================================================
    #  Step 3: solvate — 添加溶剂
    # =====================================================================

    def step_solvate(self) -> None:
        """Step 3: 添加溶剂, 溶剂模板不可用时自动尝试备选。

        遍历: spc216.gro → tip3p.gro → tip4p.gro
        每个模板最多重试 _SOLVATE_TEMPLATE_RETRY 次。
        """
        banner("Step 3/5: solvate — 添加溶剂")

        topol_path = self.result_dir / "topol.top"
        self._backup_file(topol_path)

        try:
            for template in _SOLVENT_TEMPLATES:
                for retry in range(_SOLVATE_TEMPLATE_RETRY):
                    self._count_attempt()
                    if retry > 0:
                        info(f"--- 溶剂模板 {template} 重试 {retry + 1}/{_SOLVATE_TEMPLATE_RETRY} ---")

                    t0 = time.time()
                    try:
                        self.run_gmx(
                            "solvate",
                            "-cp", f"{self.pdb_basename}_box.gro",
                            "-cs", template,
                            "-o", f"{self.pdb_basename}_solv.gro",
                            "-p", "topol.top",
                            cwd=self.result_dir,
                        )
                        dt = time.time() - t0
                        self._cleanup_backup(topol_path)
                        info(f"溶剂化完毕: {self.result_dir / f'{self.pdb_basename}_solv.gro'}")

                        self.report.add_attempt(
                            step="solvate",
                            status="success",
                            detail=f"溶剂模板={template}",
                            duration=dt,
                        )
                        return
                    except PrepError as e:
                        dt = time.time() - t0
                        warn(f"溶剂模板 '{template}' 失败 (第 {retry + 1} 次)")
                        self.report.add_attempt(
                            step="solvate",
                            status="failed",
                            detail=f"模板={template}, 重试{retry + 1}",
                            error_summary=self._extract_error_summary(e.stderr, e.stdout),
                            duration=dt,
                        )
                        # 恢复 topol.top 以便下次重试
                        self._restore_file(topol_path)
                        self._backup_file(topol_path)

            raise PrepError("solvate: 所有溶剂模板均已尝试完毕, 仍未通过")
        finally:
            self._cleanup_backup(topol_path)

    # =====================================================================
    #  Step 4: grompp (ions) — 准备离子添加输入
    # =====================================================================

    def step_grompp_ions(self) -> None:
        """Step 4: 执行 grompp 准备离子添加的 .tpr 文件。

        -maxwarn 从 10 开始递增: 10 → 20 → 50 → 100
        """
        banner("Step 4/5: grompp — 准备离子添加 (ions.tpr)")

        mdp_path = self.project_root / "mdp_files" / "ions.mdp"
        if not mdp_path.is_file():
            raise PrepError(
                f"MDP 参数文件不存在: {mdp_path}\n"
                f"请确认 mdp_files/ions.mdp 文件位于项目根目录"
            )

        for maxwarn in _MAXWARN_SEQUENCE:
            self._count_attempt()
            if maxwarn > _MAXWARN_SEQUENCE[0]:
                info(f"--- 重试: -maxwarn {maxwarn} ---")

            t0 = time.time()
            try:
                self.run_gmx(
                    "grompp",
                    "-f", str(mdp_path),
                    "-c", f"{self.pdb_basename}_solv.gro",
                    "-p", "topol.top",
                    "-o", "ions.tpr",
                    "-maxwarn", str(maxwarn),
                    cwd=self.result_dir,
                )
                dt = time.time() - t0
                info(f"grompp 完毕: {self.result_dir / 'ions.tpr'}")

                self.report.add_attempt(
                    step="grompp_ions",
                    status="success",
                    detail=f"maxwarn={maxwarn}",
                    duration=dt,
                )
                return
            except PrepError as e:
                dt = time.time() - t0
                warn(f"grompp 失败 (maxwarn={maxwarn})")
                self.report.add_attempt(
                    step="grompp_ions",
                    status="failed",
                    detail=f"maxwarn={maxwarn}",
                    error_summary=self._extract_error_summary(e.stderr, e.stdout),
                    duration=dt,
                )

        raise PrepError("grompp (ions): 所有 -maxwarn 值均已尝试完毕, 仍未通过")

    # =====================================================================
    #  Step 5: genion — 添加抗衡离子, 中和体系电荷
    # =====================================================================

    def step_genion(self) -> None:
        """Step 5: 添加 Na⁺/Cl⁻ 离子, 自动中和体系电荷。

        遍历策略 (与力场遍历模式一致):
          Phase 1: 用 "0\\n" 探测 Group 0 (System), 从 genion 输出中解析全部可用组列表。
                   识别 SOL/WAT 类型溶剂组。
          Phase 2: 构建遍历优先级 — SOL/WAT 组优先 (编号从大到小),
                   然后是非 System 非溶剂组 (编号从大到小)。
          Phase 3: 逐组遍历 genion -neutral, 直到成功或全部失败。
        """
        banner("Step 5/5: genion — 添加抗衡离子, 中和体系电荷")
        info(f"  盐浓度: {self.salt_conc} mol/L")
        info(f"  中和策略: -neutral (自动计算并中和)")

        topol_path = self.result_dir / "topol.top"

        # ---- Phase 1: 探测全部可用组 ----
        info("正在探测可用溶剂组编号 ...")
        self._backup_file(topol_path)

        all_groups: List[str] = []
        solv_group: Optional[str] = None

        t0 = time.time()
        try:
            # 显式选 Group 0 (System), 触发 genion 输出完整组列表
            self.run_gmx(
                "genion",
                "-s", "ions.tpr",
                "-o", f"{self.pdb_basename}_solv_ions.gro",
                "-p", "topol.top",
                "-pname", "NA",
                "-nname", "CL",
                "-neutral",
                "-conc", self.salt_conc,
                cwd=self.result_dir,
                stdin_input="0\n",
                capture=True,
            )
            # 意外成功 (Group 0 = System 也通过了)
            dt = time.time() - t0
            info("genion 使用组 0 (System) 意外成功")
            na_cnt, cl_cnt = _parse_ion_counts_from_topology(topol_path)
            self.solv_group_num = "0"
            self._finish_genion("0", na_cnt, cl_cnt, dt)
            return
        except PrepError as e:
            output = (e.stdout or "") + (e.stderr or "")
            all_groups = _parse_genion_group_list(output)
            solv_group = _find_solvent_group(output)

            if not all_groups:
                raise PrepError(
                    "无法从 genion 输出中解析可用组列表, "
                    "请检查 ions.tpr 是否正确生成。\n"
                    f"错误输出: {self._extract_error_summary(e.stderr, e.stdout)}"
                ) from e

            info(f"  可用组: {', '.join(all_groups)}")
            if solv_group:
                info(f"  识别到溶剂组: SOL (编号 {solv_group})")
            else:
                warn(f"  未识别到 SOL/WAT 型溶剂组, 将遍历全部非 System 组")

            self._restore_file(topol_path)

        # ---- Phase 2: 构建遍历优先级 ----
        # 非 System 组, 按编号从大到小 (编号越大是溶剂组的可能性越高)
        non_system = sorted(
            [g for g in all_groups if g != "0"],
            key=int, reverse=True,
        )

        # SOL/WAT 优先, 其余非 System 组作兜底
        candidates: List[str] = []
        if solv_group and solv_group in non_system:
            non_system.remove(solv_group)
            candidates.append(solv_group)
        candidates.extend(non_system)

        info(f"  遍历顺序: {', '.join(candidates)} ({len(candidates)} 个)")

        # ---- Phase 3: 逐组遍历 genion -neutral ----
        last_error = ""
        for idx, group_num in enumerate(candidates):
            self._count_attempt()
            info(f"  尝试组编号 {group_num} ({idx + 1}/{len(candidates)}) ...")

            # 恢复 topol.top (Phase 1 已恢复一次, 后续重试仍需恢复)
            if idx > 0:
                self._restore_file(topol_path)

            t0 = time.time()
            try:
                result = self.run_gmx(
                    "genion",
                    "-s", "ions.tpr",
                    "-o", f"{self.pdb_basename}_solv_ions.gro",
                    "-p", "topol.top",
                    "-pname", "NA",
                    "-nname", "CL",
                    "-neutral",
                    "-conc", self.salt_conc,
                    cwd=self.result_dir,
                    stdin_input=f"{group_num}\n",
                    capture=True,
                )
                dt = time.time() - t0

                if result.stdout:
                    print(result.stdout)

                self.solv_group_num = group_num
                na_cnt, cl_cnt = _parse_ion_counts_from_topology(topol_path)
                info(f"genion 使用组 {group_num} 成功")
                self._finish_genion(group_num, na_cnt, cl_cnt, dt)
                return
            except PrepError as e:
                dt = time.time() - t0
                summary = self._extract_error_summary(e.stderr, e.stdout)
                last_error = summary
                warn(f"  组 {group_num} 失败: {summary[:120]}")
                self.report.add_attempt(
                    step="genion",
                    status="failed",
                    detail=f"溶剂组={group_num}",
                    error_summary=summary,
                    duration=dt,
                )

        raise PrepError(
            f"所有 {len(candidates)} 个候选组编号均已尝试完毕, "
            f"genion 均未成功。\n"
            f"最后错误: {last_error}"
        )

    def _finish_genion(
        self, solv_group: str, na_count: int, cl_count: int, duration: float
    ) -> None:
        """记录 genion 成功完成的详细信息, 包括电荷计算结果。"""
        info(f"离子添加完毕: {self.result_dir / f'{self.pdb_basename}_solv_ions.gro'}")
        info(f"  溶剂组编号: {solv_group}")
        info(f"  添加 Na⁺:   {na_count}")
        info(f"  添加 Cl⁻:   {cl_count}")

        # 尝试计算体系净电荷
        topol_path = self.result_dir / "topol.top"
        net_charge = calculate_system_charge(topol_path)

        if net_charge is not None:
            info(f"  中和后体系净电荷: {net_charge:.4f} e")
        else:
            info(f"  体系净电荷: (无法自动解析)")

        # 电荷平衡详情
        charge_info = {
            "溶剂组编号": solv_group,
            "添加 Na⁺ 数目": na_count,
            "添加 Cl⁻ 数目": cl_count,
            "净离子添加": f"{na_count - cl_count:+d}",
            "中和后体系净电荷": f"{net_charge:.4f} e" if net_charge is not None else "未能解析",
            "盐浓度": f"{self.salt_conc} mol/L",
            "中和方式": "genion -neutral (GROMACS 自动计算并中和)",
        }
        self.report.final_params["_charge_info"] = charge_info

        self.report.add_attempt(
            step="genion",
            status="success",
            detail=(
                f"溶剂组={solv_group}, Na⁺={na_count}, Cl⁻={cl_count}, "
                f"净电荷={net_charge:.4f}" if net_charge is not None else
                f"溶剂组={solv_group}, Na⁺={na_count}, Cl⁻={cl_count}"
            ),
            duration=duration,
        )

    # =====================================================================
    #  文件收集
    # =====================================================================

    def _collect_generated_files(self) -> None:
        """扫描输出目录, 收集所有生成的文件并记录到报告。"""
        collected: List[str] = []

        # 核心文件 (按重要性排序)
        key_files = [
            "topol.top",
            f"{self.pdb_basename}_solv_ions.gro",
            f"{self.pdb_basename}_solv.gro",
            f"{self.pdb_basename}_box.gro",
            f"{self.pdb_basename}_processed.gro",
            "ions.tpr",
            "posre.itp",
        ]
        for fname in key_files:
            fpath = self.result_dir / fname
            if fpath.is_file():
                size_kb = fpath.stat().st_size / 1024
                collected.append(f"{fname:<35s} {size_kb:>8.1f} KB")

        # 其余 .itp 文件
        for itp in sorted(self.result_dir.glob("*.itp")):
            if itp.name not in key_files:
                size_kb = itp.stat().st_size / 1024
                collected.append(f"{itp.name:<35s} {size_kb:>8.1f} KB")

        self.report.generated_files = collected

    # =====================================================================
    #  前置检查
    # =====================================================================

    def _check_gmx(self) -> None:
        """验证 GROMACS 是否可用。"""
        info("检测 GROMACS 环境 ...")
        try:
            result = self.run_gmx("--version", capture=True)
            version_line = (result.stdout or "").strip().split("\n")[0]
            info(f"GROMACS 版本: {version_line}")
        except PrepError:
            raise PrepError(
                f"无法运行 '{self.gmx_cmd}' 命令。"
                f"请确认 GROMACS 已安装并激活 (conda activate / source GMXRC)。"
            ) from None

    def _check_input(self) -> None:
        """验证输入 PDB 文件存在。"""
        pdb_full = self.project_root / self.pdb_path
        if not pdb_full.is_file():
            if self.dry_run:
                warn(f"输入文件不存在 (dry-run 模式, 跳过检查): {pdb_full}")
                return
            raise PrepError(f"输入 PDB 文件不存在: {pdb_full}")
        info(f"输入 PDB: {self.pdb_path}")
        info(f"体系名称: {self.pdb_basename}")

    def _setup_dirs(self) -> None:
        """创建输出目录。"""
        if not self.dry_run:
            self.result_dir.mkdir(parents=True, exist_ok=True)
        info(f"输出目录: {self.result_dir}")

    # =====================================================================
    #  主流程
    # =====================================================================

    def run(self) -> dict:
        """执行全部 5 个准备步骤, 返回结果字典。

        Returns:
            dict:
                success           — bool, 是否全部成功
                forcefield_num    — str, 最终力场编号
                forcefield_name   — str, 最终力场名
                water_num         — str, 最终水模型编号
                water_name        — str, 最终水模型名
                box_type          — str, 盒子类型
                result_dir        — str, 输出目录路径
                report_path       — str, 遍历报告路径
                total_attempts    — int, 总命令尝试次数
                error             — str, 错误信息 (仅失败时)
        """
        banner("GROMACS MD 准备步骤 — 自动遍历脚本")
        info(f"输入: {self.pdb_path}")

        try:
            # ---- 前置检查 ----
            self._check_gmx()
            self._check_input()
            self._setup_dirs()

            # ---- 执行步骤 ----
            self.step_pdb2gmx()      # 1: 拓扑生成 (力场×水模型遍历)
            self.step_editconf()     # 2: 定义盒子
            self.step_solvate()      # 3: 添加溶剂
            self.step_grompp_ions()  # 4: 准备离子添加
            self.step_genion()       # 5: 离子中和

            # ---- 成功收尾 ----
            self.report.success = True
            self.report.final_params.update({
                "力场编号": self.forcefield_num,
                "力场名称": FORCEFIELD_MAP[self.forcefield_num][1],
                "力场短名": self.forcefield_name,
                "水模型编号": self.water_num,
                "水模型名称": WATER_MAP[self.water_num][1],
                "水模型短名": self.water_name,
                "盒子类型": self.box_type,
                "盒子边缘距离": f"{self.box_dist} nm",
                "盐浓度": f"{self.salt_conc} mol/L",
                "总命令尝试次数": self._total_attempts,
            })

            self._collect_generated_files()
            report_path = self.report.save()

            banner("全部准备步骤完成")
            info(f"遍历报告: {report_path}")
            info(f"共进行 {self._total_attempts} 次命令尝试")

            return {
                "success": True,
                "forcefield_num": self.forcefield_num,
                "forcefield_name": self.forcefield_name,
                "water_num": self.water_num,
                "water_name": self.water_name,
                "box_type": self.box_type,
                "result_dir": str(self.result_dir),
                "report_path": str(report_path),
                "total_attempts": self._total_attempts,
            }

        except PrepError as e:
            self.report.success = False
            self.report.final_params["错误信息"] = str(e)
            self._collect_generated_files()
            report_path = self.report.save()

            banner("准备步骤失败")
            error(str(e))
            info(f"遍历报告 (含失败记录): {report_path}")

            return {
                "success": False,
                "error": str(e),
                "result_dir": str(self.result_dir),
                "report_path": str(report_path),
                "total_attempts": self._total_attempts,
            }

        except KeyboardInterrupt:
            self.report.success = False
            self.report.final_params["错误信息"] = "用户中断 (Ctrl+C)"
            self._collect_generated_files()
            report_path = self.report.save()

            warn("\n用户中断")
            info(f"遍历报告: {report_path}")

            return {
                "success": False,
                "error": "用户中断",
                "result_dir": str(self.result_dir),
                "report_path": str(report_path),
            }


# =========================================================================
#  公共 API — 供 run_md.py 通过 import 调用
# =========================================================================

def run_preparation(
    pdb_path: str,
    result_dir: str = "MD_result/pdb2gmx",
    gmx_cmd: str = "gmx",
    box_type: str = "dodecahedron",
    box_dist: str = "1.0",
    salt_conc: str = "0.15",
    dry_run: bool = False,
) -> dict:
    """执行 GROMACS MD 全部准备步骤 (力场遍历 + 电荷中和)。

    自动遍历力场×水模型组合, 生成拓扑文件、溶剂化结构,
    并通过 genion -neutral 自动中和体系电荷。

    Args:
        pdb_path:   输入 PDB 文件路径 (相对于当前工作目录或绝对路径)
        result_dir: 输出目录, 默认 "MD_result"
        gmx_cmd:    GROMACS 可执行文件, 默认 "gmx"
        box_type:   盒子类型 (dodecahedron / cubic / octahedron / triclinic)
        box_dist:   盒子边缘到蛋白最短距离 (nm 字符串)
        salt_conc:  盐浓度 (mol/L 字符串)
        dry_run:    True 时只打印命令不执行

    Returns:
        dict:
            success         — bool
            forcefield_num  — str  最终力场编号
            forcefield_name — str  最终力场短名
            water_num       — str  最终水模型编号
            water_name      — str  最终水模型短名
            box_type        — str  盒子类型
            result_dir      — str  输出目录
            report_path     — str  traverse_report.txt 绝对路径
            total_attempts  — int  命令执行总次数
            error           — str  (仅失败时)

    Example:
        >>> import importlib
        >>> prep = importlib.import_module("01_pdb2gmx")
        >>> result = prep.run_preparation("input_rename/protein.pdb")
        >>> print(result["forcefield_name"])
        charmm36

    在 run_md.py 中集成示例:
        # 在 run_md.py 的 step_pdb2gmx ~ step_genion 位置:
        import importlib
        prep = importlib.import_module("01_pdb2gmx")
        result = prep.run_preparation(
            pdb_path=self.input_pdb,
            result_dir=str(self.result_dir),
        )
        if not result["success"]:
            raise MDError(1, stderr=result.get("error", ""))
        # 同步参数回 run_md.py 的 runner
        self.forcefield_num = result["forcefield_num"]
        self.forcefield_name = result["forcefield_name"]
        self.water_num = result["water_num"]
        self.water_name = result["water_name"]
        self.box_type = result["box_type"]
    """
    runner = PreparationRunner(
        pdb_path=pdb_path,
        result_dir=result_dir,
        gmx_cmd=gmx_cmd,
        box_type=box_type,
        box_dist=box_dist,
        salt_conc=salt_conc,
        dry_run=dry_run,
    )
    return runner.run()


# =========================================================================
#  CLI 入口 — 作为独立脚本运行
# =========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="01_pdb2gmx.py",
        description=(
            "GROMACS MD 准备步骤自动化 — "
            "自动遍历力场/水模型, 添加离子中和电荷"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 01_pdb2gmx.py --pdb input_rename/protein.pdb
  python 01_pdb2gmx.py --pdb input_pdb/2UWF.pdb --output my_MD
  python 01_pdb2gmx.py --pdb protein.pdb --dry-run
  python 01_pdb2gmx.py --pdb protein.pdb --box-type cubic --salt-conc 0.2
  python 01_pdb2gmx.py --pdb protein.pdb --box-dist 1.2
        """,
    )

    # 必选参数
    parser.add_argument(
        "--pdb", required=True,
        help="输入 PDB 文件路径 (相对于当前工作目录或绝对路径)",
    )

    # 可选参数
    parser.add_argument(
        "--output", default=DEFAULTS["result_dir"],
        help=f"输出目录 (默认: {DEFAULTS['result_dir']})",
    )
    parser.add_argument(
        "--gmx", default=DEFAULTS["gmx"],
        help=f"GROMACS 命令 (默认: {DEFAULTS['gmx']})",
    )
    parser.add_argument(
        "--box-type", default=DEFAULTS["box_type"],
        choices=["cubic", "dodecahedron", "triclinic", "octahedron"],
        help=f"盒子类型 (默认: {DEFAULTS['box_type']})",
    )
    parser.add_argument(
        "--box-dist", default=DEFAULTS["box_dist"],
        help=f"盒子边缘距离 (nm, 默认: {DEFAULTS['box_dist']})",
    )
    parser.add_argument(
        "--salt-conc", default=DEFAULTS["salt_conc"],
        help=f"盐浓度 (mol/L, 默认: {DEFAULTS['salt_conc']})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="只打印命令不执行 (预览模式)",
    )

    args = parser.parse_args(argv)

    result = run_preparation(
        pdb_path=args.pdb,
        result_dir=args.output,
        gmx_cmd=args.gmx,
        box_type=args.box_type,
        box_dist=args.box_dist,
        salt_conc=args.salt_conc,
        dry_run=args.dry_run,
    )

    if result["success"]:
        print(f"\n{'=' * 60}")
        print(f"  准备步骤全部完成")
        print(f"  力场:     {result['forcefield_name']}")
        print(f"  水模型:   {result['water_name']}")
        print(f"  盒子类型: {result['box_type']}")
        print(f"  遍历报告: {result['report_path']}")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"  准备步骤失败")
        print(f"  原因: {result.get('error', '未知')}")
        print(f"  遍历报告: {result.get('report_path', 'N/A')}")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
