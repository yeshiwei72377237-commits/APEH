#!/usr/bin/env python3
"""
GROMACS NPT 压力平衡 (恒温恒压平衡) 自动化脚本
自动控制 NPT 步骤, 检测温度/压力/密度收敛质量, 不收敛时自动调整参数重试

检测策略:
  - 解析 mdrun 输出, 检查温度是否接近目标值 (ref_t) 并稳定
  - 检查压力是否接近目标值 (ref_p) 并稳定
  - 检查密度是否收敛 (无明显漂移)
  - 分析势能 (Epot) 变化曲线, 确认无明显漂移
  - 通过 gmx energy 提取温度/压力/势能/密度时间序列进行定量分析
  - 检测致命错误: LINCS WARNING / nan / inf / step size too large

不收敛时自动调整参数 (在原 mdp 文件外创建新文件, 原始 npt.mdp 不被修改):
  dt:     0.002 → 0.001 → 0.0005
  nsteps: 50000 → 100000 → 200000 → 500000
  tau_p:  2.0 → 1.0 → 0.5
  tau_t:  0.1 → 0.05

输入:  MD_result/nvt/       (从 NVT 步骤读取 nvt.gro + nvt.cpt, 从上级目录读取 topol.top)
输出:  MD_result/npt/
  ├── npt.tpr / npt.gro / npt.log / npt.edr / npt.trr / npt.cpt / npt.xtc
  ├── adjusted_mdp/           # 调整后的参数文件 (原始 npt.mdp 不被修改)
  │   ├── npt_attempt_1.mdp
  │   └── ...
  └── traverse_report.txt     # 遍历报告

用法:
  # 独立脚本
  python 04_npt.py

  # 被 run_md.py 导入
  import importlib
  npt = importlib.import_module("04_npt")
  result = npt.run_npt()
"""

from __future__ import annotations

import re
import subprocess
import sys

from scripts._gmx_utils import run_gmx
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


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


# ==================== 默认参数 ====================

DEFAULTS = {
    "input_dir": "MD_result/nvt",
    "output_dir": "MD_result/npt",
    "gmx": "gmx",
    "dry_run": False,
}

# MDP 模板路径 (项目根目录下)
_MDP_TEMPLATE_DIR = "mdp_files"
_NPT_MDP_TEMPLATE = "npt.mdp"

# grompp -maxwarn 递增序列
_MAXWARN_SEQUENCE: List[int] = [10, 20, 50, 100]

# GPU → CPU 降级重试
_MDRUN_GPU_RETRY = 3
_MDRUN_CPU_RETRY = 3

# 全局最大尝试次数
_MAX_TOTAL_ATTEMPTS = 100


# ==================== NPT 参数遍历策略 ====================

# 不收敛时的 dt + nsteps + tau_p + tau_t 组合遍历顺序
# 原则: 先增加步数, 再收紧压力耦合, 再减半步长, 最后收紧温度耦合
_NPT_PARAM_TRAVERSE: List[Tuple[float, int, float, float, str]] = [
    # (dt, nsteps, tau_p, tau_t, 说明)
    (0.002, 50000,  2.0, 0.1,  "默认参数"),
    (0.002, 100000, 2.0, 0.1,  "增加步数 ×2"),
    (0.002, 200000, 2.0, 0.1,  "增加步数 ×4"),
    (0.002, 200000, 1.0, 0.1,  "增加步数 + 收紧压力耦合"),
    (0.001, 200000, 1.0, 0.1,  "减半步长 + 增加步数 + 收紧压力"),
    (0.001, 200000, 1.0, 0.05, "减半步长 + 增加步数 + 收紧温压耦合"),
    (0.0005, 500000, 0.5, 0.05,"最严格: 小步长 + 长平衡 + 紧耦合"),
]


# ==================== 异常类 ====================

class NPTError(Exception):
    """NPT 步骤中的错误。"""

    def __init__(self, message: str, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


class NPTConfigError(NPTError):
    """配置/解析类错误 — 重试阶梯 (dt/nsteps/tau_p/tau_t) 无法改善, 应立即停止。"""


# ==================== NPT 质量检测 ====================

class NPTQualityResult:
    """NPT 压力平衡质量检测结果。"""

    def __init__(self) -> None:
        self.equilibrated: bool = False
        # 温度
        self.temp_average: Optional[float] = None       # 平均温度 (K)
        self.temp_target: Optional[float] = None         # 目标温度 (K)
        self.temp_rmsd: Optional[float] = None           # 温度波动 RMSD (K)
        self.temp_drift: Optional[float] = None          # 温度漂移 (K)
        # 压力
        self.pres_average: Optional[float] = None        # 平均压力 (bar)
        self.pres_target: Optional[float] = None          # 目标压力 (bar)
        self.pres_rmsd: Optional[float] = None            # 压力波动 RMSD (bar)
        self.pres_drift: Optional[float] = None           # 压力漂移 (bar)
        # 密度
        self.dens_average: Optional[float] = None         # 平均密度 (kg/m³)
        self.dens_drift_percent: Optional[float] = None   # 密度漂移百分比
        # 势能
        self.pot_average: Optional[float] = None          # 平均势能
        self.pot_initial: Optional[float] = None          # 初始势能 (前10%平均)
        self.pot_final: Optional[float] = None            # 最终势能 (后10%平均)
        self.pot_drift_percent: Optional[float] = None    # 势能漂移百分比
        # 时间序列
        self.temp_values: List[float] = []                # 温度时间序列
        self.pres_values: List[float] = []                # 压力时间序列
        self.pot_values: List[float] = []                 # 势能时间序列
        self.dens_values: List[float] = []                # 密度时间序列
        # 模拟信息
        self.total_steps: int = 0
        self.dt: Optional[float] = None                   # 时间步长 (ps)
        self.sim_time: Optional[float] = None             # 模拟时长 (ps)
        # 致命错误
        self.has_lincs_warning: bool = False
        self.has_nan_inf: bool = False
        self.has_step_size_error: bool = False
        # 错误收集
        self.errors: List[str] = []


def _parse_npt_log(log_path: Path) -> NPTQualityResult:
    """解析 GROMACS NPT 的 .log 文件, 检测致命错误和提取模拟参数。

    注意: 温度/压力/密度/势能的质量分析已移至 _analyze_edr_quality(),
    通过 gmx energy 从 .edr 文件提取。本函数仅负责:
      1. 致命错误检测: LINCS WARNING / nan / inf / step size too large
      2. 模拟参数提取: dt, nsteps, ref_t, ref_p
      3. 步进温度/压力快照提取 (辅助参考)

    Args:
        log_path: npt.log 文件路径

    Returns:
        NPTQualityResult (equilibrated 仅反映致命错误状态,
        温度/压力/密度/势能分析结果由 _analyze_edr_quality 补充)
    """
    result = NPTQualityResult()

    if not log_path.is_file():
        result.errors.append("日志文件不存在")
        return result

    content = log_path.read_text(encoding="utf-8", errors="replace")

    # ---- 1. 致命错误检测 ----

    if re.search(r'LINCS\s+WARNING', content, re.IGNORECASE):
        result.has_lincs_warning = True
        result.errors.append("LINCS WARNING — 约束算法失败, 体系中存在严重冲突")

    # nan/inf 检测: 只在模拟输出段搜索
    # (参数段含 "epsilon-rf = inf" 等合法值, 不能盲目全文搜索)
    _step_lines = content.splitlines()
    _step_start = 0
    for i, line in enumerate(_step_lines):
        if re.search(r'^\s*Step\s{2,}', line):
            _step_start = i
            break
    _step_content = "\n".join(_step_lines[_step_start:]) if _step_start > 0 else content

    if re.search(r'\bnan\b', _step_content, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("检测到 nan 值 — 体系中存在非物理接触或参数错误")
    if re.search(r'\binf\b', _step_content, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("检测到 inf 值 — 体系中存在非物理接触或参数错误")

    if re.search(r'step\s+size\s+too\s+large', content, re.IGNORECASE):
        result.has_step_size_error = True
        result.errors.append("Step size too large — 力场梯度过大, 体系不稳定")

    # ---- 2. 提取模拟参数 ----

    # dt
    dt_match = re.search(r'(?:^|\n)\s*dt\s*=\s*([\d.]+)', content, re.IGNORECASE)
    if dt_match:
        result.dt = float(dt_match.group(1))

    # nsteps
    nsteps_match = re.search(r'(?:^|\n)\s*nsteps\s*=\s*(\d+)', content, re.IGNORECASE)
    if nsteps_match:
        result.total_steps = int(nsteps_match.group(1))

    # ref_t — 可能有多个值 (多组耦合), 取第一个
    ref_t_match = re.search(
        r'(?:^|\n)\s*ref[-_]?t\s*[:=]\s*([\d.]+(?:\s+[\d.]+)*)',
        content, re.IGNORECASE,
    )
    if ref_t_match:
        ref_t_str = ref_t_match.group(1).strip()
        result.temp_target = float(ref_t_str.split()[0])

    # ref_p — 可能有多个值, 取第一个
    ref_p_match = re.search(
        r'(?:^|\n)\s*ref[-_]?p\s*[:=]\s*([\d.]+(?:\s+[\d.]+)*)',
        content, re.IGNORECASE,
    )
    if ref_p_match:
        ref_p_str = ref_p_match.group(1).strip()
        result.pres_target = float(ref_p_str.split()[0])

    # ---- 3. 解析每步温度和压力 ----
    # GROMACS MD .log 中的输出格式:
    #   新版 (列式输出):
    #      Temperature    Pressure
    #      300.123        1.234
    #   旧版:
    #      Temperature
    #      300.123
    #
    # NPT 输出通常每步一行, 包含多个物理量

    lines = content.splitlines()

    # 尝试列式格式: 查找包含 "Temperature" 和 "Pressure" 的标题行
    # 典型格式 (GROMACS 新版):
    #    Step           Time         Temperature    Pressure       Density
    #       0        0.0000          300.00000        1.00000     1000.000
    temp_vals: List[float] = []
    pres_vals: List[float] = []
    dens_vals: List[float] = []

    # 方式 A: 解析列式表格 (每行 Step ... Temperature ... Pressure ... Density)
    col_header_pattern = re.compile(
        r'Step\s+Time\s+.*(?:Temperature|Temp\.)',
        re.IGNORECASE,
    )
    in_table = False
    for line in lines:
        stripped = line.strip()
        if col_header_pattern.search(stripped):
            in_table = True
            continue
        if in_table:
            if not stripped:
                in_table = False
                continue
            # 尝试按空白分割并解析数值
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    # 通常格式: Step Time Temperature Pressure [Density ...]
                    # 找到合理范围内的值
                    for p in parts:
                        try:
                            val = float(p)
                            # 温度范围 0~2000K
                            if 250.0 < val < 500.0 and len(temp_vals) == len(pres_vals):
                                temp_vals.append(val)
                                continue
                            # 压力范围 -500~5000 bar (NPT 波动较大)
                            if -500.0 < val < 5000.0 and len(pres_vals) < len(temp_vals):
                                pres_vals.append(val)
                                continue
                        except ValueError:
                            pass
                except (ValueError, IndexError):
                    pass

    # 方式 B: 如果列式解析失败, 尝试旧版逐个 section 格式
    if not temp_vals:
        temp_section_pattern = re.compile(
            r'^\s*Temperature\s*$',
            re.MULTILINE | re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            if temp_section_pattern.match(line.strip()):
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        break
                    if next_line.startswith("Statistics") or next_line.startswith("Energies"):
                        break
                    parts = next_line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if 0.0 < val < 2000.0:
                                temp_vals.append(val)
                        except ValueError:
                            pass

    # 方式 C: 解析旧版 Pressure section
    if not pres_vals:
        pres_section_pattern = re.compile(
            r'^\s*Pressure\s*$',
            re.MULTILINE | re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            if pres_section_pattern.match(line.strip()):
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        break
                    if next_line.startswith("Statistics") or next_line.startswith("Energies"):
                        break
                    parts = next_line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if -500.0 < val < 5000.0:
                                pres_vals.append(val)
                        except ValueError:
                            pass

    result.temp_values = temp_vals
    result.pres_values = pres_vals

    # ---- 4. 计算模拟时长 ----
    if result.dt is not None and result.total_steps > 0:
        result.sim_time = result.dt * result.total_steps

    # ---- 5. 基于步进值的快速度估算 (辅助参考) ----
    if temp_vals and len(temp_vals) >= 10:
        n_temp = len(temp_vals)
        front_n = max(1, n_temp // 5)
        front_avg = sum(temp_vals[:front_n]) / front_n
        back_avg = sum(temp_vals[-front_n:]) / front_n
        result.temp_drift = back_avg - front_avg

    if pres_vals and len(pres_vals) >= 10:
        n_pres = len(pres_vals)
        front_n = max(1, n_pres // 5)
        front_avg = sum(pres_vals[:front_n]) / front_n
        back_avg = sum(pres_vals[-front_n:]) / front_n
        result.pres_drift = back_avg - front_avg

    if dens_vals and len(dens_vals) >= 10:
        n_dens = len(dens_vals)
        front_n = max(1, n_dens // 5)
        front_avg = sum(dens_vals[:front_n]) / front_n
        back_avg = sum(dens_vals[-front_n:]) / front_n
        result.dens_average = sum(dens_vals) / n_dens
        if abs(front_avg) > 1e-10:
            result.dens_drift_percent = (back_avg - front_avg) / abs(front_avg) * 100

    # ---- 6. 仅基于致命错误的判定 ----
    # 注意: 温度/压力/密度/势能的质量分析由 _analyze_edr_quality() 从 .edr 文件完成
    # 这里仅检查致命错误 (LINCS/nan/inf/step size), 不做热力学量判定
    result.equilibrated = (
        not result.has_lincs_warning
        and not result.has_nan_inf
        and not result.has_step_size_error
    )

    return result


# ==================== 能量文件解析辅助 ====================

def _extract_energy_series(
    edr_path: Path,
    gmx_cmd: str,
    term: str,
    dry_run: bool = False,
) -> Tuple[List[float], List[float]]:
    """使用 gmx energy 从 .edr 文件提取能量/温度/压力/密度时间序列。

    GROMACS 官方推荐方式 — 输出格式跨版本稳定, 不受 .log 文件格式变化影响。
    使用 -xvg none 获得纯数值输出, 唯一临时文件名避免多次调用互相覆盖。

    Args:
        edr_path: .edr 能量文件路径
        gmx_cmd:  GROMACS 命令
        term:     要提取的项名称 (如 "Temperature", "Pressure", "Potential", "Density")
        dry_run:  预览模式

    Returns:
        (time_list, value_list) — 时间和值的时间序列
    """
    if not edr_path.is_file():
        return [], []

    if dry_run:
        return [], []

    # 唯一临时文件名 — 多次提取不会互相覆盖
    safe_term = re.sub(r'[^a-zA-Z0-9]', '_', term)
    xvg_name = f"_temp_npt_{safe_term}.xvg"
    xvg_path = edr_path.parent / xvg_name

    try:
        run_gmx(
            [gmx_cmd, "energy", "-f", str(edr_path), "-o", str(xvg_path), "-xvg", "none"],
            input=f"{term}\n0\n",
            capture_output=True,
            cwd=str(edr_path.parent),
            check=True,
        )

        if not xvg_path.is_file():
            return [], []

        xvg_content = xvg_path.read_text(encoding="utf-8", errors="replace")
        times: List[float] = []
        values: List[float] = []
        for line in xvg_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue

        # 清理临时文件
        try:
            xvg_path.unlink()
        except OSError:
            pass

        return times, values

    except (subprocess.CalledProcessError, FileNotFoundError):
        return [], []


# ==================== EDR 能量文件质量分析 (核心) ====================

def _analyze_edr_quality(
    edr_path: Path,
    gmx_cmd: str,
    quality: NPTQualityResult,
    dry_run: bool = False,
    temp_target: Optional[float] = None,
    pres_target: Optional[float] = None,
) -> None:
    """使用 gmx energy 从 .edr 文件提取温度/压力/密度/势能时间序列并分析平衡质量。

    这是 NPT 质量检测的核心函数。与 _parse_npt_log() 分工:
      - _parse_npt_log()      → 致命错误检测 + 模拟参数提取 (dt, nsteps, ref_t, ref_p)
      - _analyze_edr_quality() → 温度/压力/密度/势能时间序列分析 + 收敛判定

    为什么用 gmx energy 而非解析 .log 的 Statistics 块?
      - GROMACS Statistics 块格式因版本而异 (列顺序、科学记数法、表头行)
      - 正则跨行匹配极易失效 (缺少 re.DOTALL、列偏移等)
      - gmx energy 是 GROMACS 官方工具, 输出 `-xvg none` 后为固定两列数值,
        跨版本完全稳定

    直接修改传入的 NPTQualityResult 对象 (原地修改)。

    Args:
        edr_path: npt.edr 能量文件路径
        gmx_cmd:  GROMACS 命令
        quality:  待更新的质量检测结果 (原地修改)
        dry_run:  预览模式
        temp_target: 目标温度 (K) — 由调用方从实际使用的 mdp 中读取并传入
        pres_target: 目标压力 (bar) — 同上
    """
    if dry_run:
        return

    # 目标值以调用方传入的实际 mdp 为准 (日志解析仅作兜底)
    if temp_target is not None:
        quality.temp_target = temp_target
    if pres_target is not None:
        quality.pres_target = pres_target

    if not edr_path.is_file():
        quality.errors.append("能量文件 (.edr) 不存在, 无法进行热力学量分析")
        quality.equilibrated = False
        return

    # =====================================================================
    #  提取温度时间序列
    # =====================================================================

    _, temp_vals = _extract_energy_series(
        edr_path, gmx_cmd, "Temperature", dry_run,
    )

    if temp_vals:
        quality.temp_values = temp_vals
        n = len(temp_vals)

        # 平均温度
        quality.temp_average = sum(temp_vals) / n

        # 温度 RMSD
        variance = sum((v - quality.temp_average) ** 2 for v in temp_vals) / n
        quality.temp_rmsd = variance ** 0.5

        # 温度漂移: 后 20% 段 与 前 20% 段的差异
        if n >= 10:
            front_n = max(1, n // 5)
            front_avg = sum(temp_vals[:front_n]) / front_n
            back_avg = sum(temp_vals[-front_n:]) / front_n
            quality.temp_drift = back_avg - front_avg

        # ---- 温度质量判断 ----
        if quality.temp_target is None:
            # 目标温度必须由调用方 (mdp) 或日志提供 — 缺失时不得静默通过
            quality.errors.append(
                "无法确定目标温度: mdp 中缺少 ref-t / gen-temp, 日志中也未解析到"
            )
        if quality.temp_target is not None:
            temp_diff = abs(quality.temp_average - quality.temp_target)

            if temp_diff > 20.0:
                quality.errors.append(
                    f"温度偏差过大: 平均={quality.temp_average:.1f}K, "
                    f"目标={quality.temp_target:.1f}K, 偏差={temp_diff:.1f}K"
                )

        if quality.temp_rmsd is not None and quality.temp_rmsd > 20.0:
            quality.errors.append(
                f"温度波动过大: RMSD={quality.temp_rmsd:.1f}K (阈值 20.0K)"
            )

        if quality.temp_drift is not None and abs(quality.temp_drift) > 40.0:
            quality.errors.append(
                f"温度漂移过大: {quality.temp_drift:+.1f}K "
                f"(前段→后段差异 > 40K, 体系可能未充分平衡)"
            )
    else:
        quality.errors.append("无法从能量文件中提取温度数据, 请检查模拟是否正常完成")

    # =====================================================================
    #  提取压力时间序列
    # =====================================================================

    _, pres_vals = _extract_energy_series(
        edr_path, gmx_cmd, "Pressure", dry_run,
    )

    if pres_vals:
        quality.pres_values = pres_vals
        n = len(pres_vals)

        # 平均压力
        quality.pres_average = sum(pres_vals) / n

        # 压力 RMSD (NPT 压力瞬时波动很大是正常的)
        variance = sum((v - quality.pres_average) ** 2 for v in pres_vals) / n
        quality.pres_rmsd = variance ** 0.5

        # 压力漂移: 后 20% 段 与 前 20% 段的差异
        if n >= 10:
            front_n = max(1, n // 5)
            front_avg = sum(pres_vals[:front_n]) / front_n
            back_avg = sum(pres_vals[-front_n:]) / front_n
            quality.pres_drift = back_avg - front_avg

        # ---- 压力质量判断 ----
        if quality.pres_target is None:
            # 目标压力必须由调用方 (mdp) 或日志提供 — 缺失时不得静默通过
            quality.errors.append(
                "无法确定目标压力: mdp 中缺少 ref-p, 日志中也未解析到"
            )
        if quality.pres_target is not None:
            pres_diff = abs(quality.pres_average - quality.pres_target)

            if pres_diff > 100.0:
                quality.errors.append(
                    f"压力偏差过大: 平均={quality.pres_average:.1f}bar, "
                    f"目标={quality.pres_target:.1f}bar, 偏差={pres_diff:.1f}bar"
                )

        if quality.pres_rmsd is not None and quality.pres_rmsd > 200.0:
            quality.errors.append(
                f"压力波动过大: RMSD={quality.pres_rmsd:.1f}bar (阈值 200.0bar)"
            )

        if quality.pres_drift is not None and abs(quality.pres_drift) > 200.0:
            quality.errors.append(
                f"压力漂移过大: {quality.pres_drift:+.1f}bar "
                f"(前段→后段差异 > 200bar)"
            )
    else:
        quality.errors.append("无法从能量文件中提取压力数据, 请检查模拟是否正常完成")

    # =====================================================================
    #  提取密度时间序列
    # =====================================================================

    _, dens_vals = _extract_energy_series(
        edr_path, gmx_cmd, "Density", dry_run,
    )

    if dens_vals:
        quality.dens_values = dens_vals
        n = len(dens_vals)

        # 平均密度
        quality.dens_average = sum(dens_vals) / n

        # 密度漂移: 后 20% 段 与 前 20% 段的差异
        if n >= 10:
            front_n = max(1, n // 5)
            front_avg = sum(dens_vals[:front_n]) / front_n
            back_avg = sum(dens_vals[-front_n:]) / front_n
            if abs(front_avg) > 1e-10:
                quality.dens_drift_percent = (back_avg - front_avg) / abs(front_avg) * 100
                if abs(quality.dens_drift_percent) > 10.0:
                    quality.errors.append(
                        f"密度漂移过大: {quality.dens_drift_percent:+.1f}% "
                        f"(前后 20% 段差异 > 10%)"
                    )

    # =====================================================================
    #  提取势能时间序列
    # =====================================================================

    _, pot_vals = _extract_energy_series(
        edr_path, gmx_cmd, "Potential", dry_run,
    )

    if pot_vals:
        quality.pot_values = pot_vals
        n = len(pot_vals)

        # 平均势能
        quality.pot_average = sum(pot_vals) / n

        # 势能漂移: 后 20% 段 与 前 20% 段的差异
        if n >= 10:
            front_n = max(1, n // 5)
            quality.pot_initial = sum(pot_vals[:front_n]) / front_n
            quality.pot_final = sum(pot_vals[-front_n:]) / front_n

            if quality.pot_initial and abs(quality.pot_initial) > 1e-10:
                drift = (
                    (quality.pot_final - quality.pot_initial)
                    / abs(quality.pot_initial) * 100
                )
                quality.pot_drift_percent = drift
                if abs(drift) > 20.0:
                    quality.errors.append(
                        f"势能漂移过大: {drift:+.1f}% (前后 20% 段差异 > 20%)"
                    )

    # =====================================================================
    #  综合判定
    # =====================================================================
    # equilibrated 需要同时满足:
    #   1. 无致命错误 (_parse_npt_log 已检查, 记录在 quality.errors / flags 中)
    #   2. 无温度/压力/密度/势能问题 (本函数添加的错误也在 quality.errors 中)
    quality.equilibrated = (
        not quality.has_lincs_warning
        and not quality.has_nan_inf
        and not quality.has_step_size_error
        and len(quality.errors) == 0
    )


# ==================== MDP 参数文件管理 ====================

# mdp 中目标值行: 支持 ref-t / ref_t / ref t 等写法 (GROMACS 日志写成 "ref-t:" 也可解析)
_MDP_REF_T_RE = re.compile(
    r'(?:^|\n)\s*ref[-_]?t\s*[:=]\s*([\d.]+)',
    re.IGNORECASE,
)
_MDP_REF_P_RE = re.compile(
    r'(?:^|\n)\s*ref[-_]?p\s*[:=]\s*([\d.]+)',
    re.IGNORECASE,
)
# ref-t 缺失时的退路: gen-temp
_MDP_GEN_TEMP_RE = re.compile(
    r'(?:^|\n)\s*gen[-_]?temp\s*[:=]\s*([\d.]+)',
    re.IGNORECASE,
)


def _read_mdp_targets(mdp_text: str) -> Tuple[Optional[float], Optional[float]]:
    """从 mdp 文本中读取 (目标温度 K, 目标压力 bar)。

    温度优先级: ref-t > gen-temp; 压力: ref-p。
    读取不到时对应项返回 None — 调用方必须视为配置错误并终止,
    不要用硬编码默认值 (如 300.0 K / 1.0 bar) 兜底。
    """
    temp_target: Optional[float] = None
    match = _MDP_REF_T_RE.search(mdp_text)
    if match:
        temp_target = float(match.group(1))
    else:
        match = _MDP_GEN_TEMP_RE.search(mdp_text)
        if match:
            temp_target = float(match.group(1))

    pres_target: Optional[float] = None
    match = _MDP_REF_P_RE.search(mdp_text)
    if match:
        pres_target = float(match.group(1))

    return temp_target, pres_target


def _read_template_mdp(project_root: Path) -> str:
    """读取原始 npt.mdp 模板内容。"""
    template_path = project_root / _MDP_TEMPLATE_DIR / _NPT_MDP_TEMPLATE
    if not template_path.is_file():
        raise NPTConfigError(
            f"NPT 参数模板文件不存在: {template_path}\n"
            f"请确认 {_MDP_TEMPLATE_DIR}/{_NPT_MDP_TEMPLATE} 位于项目根目录"
        )
    return template_path.read_text(encoding="utf-8")


def _create_adjusted_mdp(
    template_content: str,
    dt: float,
    nsteps: int,
    tau_p: float,
    tau_t: float,
    output_path: Path,
) -> Path:
    """基于模板内容创建调整后的 mdp 文件 (不修改原始文件)。

    Args:
        template_content: 原始 npt.mdp 的文本内容
        dt:              新的时间步长 (ps)
        nsteps:          新的步数
        tau_p:           新的压力耦合时间常数
        tau_t:           新的温度耦合时间常数
        output_path:     输出文件路径

    Returns:
        创建的文件路径
    """
    content = template_content

    # 替换 dt 行
    content = re.sub(
        r'^(\s*dt\s*=\s*)(.*)$',
        lambda m: f'dt          = {dt:.4f}'.rstrip('0').rstrip('.')
        if '.' in f'{dt}' else f'dt          = {dt}',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 替换 nsteps 行
    content = re.sub(
        r'^(\s*nsteps\s*=\s*)(.*)$',
        f'nsteps      = {nsteps}',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 替换 tau-p 行
    content = re.sub(
        r'^(\s*tau-p\s*=\s*)(.*)$',
        f'tau-p       = {tau_p:.1f}',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 替换 tau-t 行 (可能有多个值)
    content = re.sub(
        r'^(\s*tau-t\s*=\s*)(.*)$',
        lambda m: f'tau-t       = {tau_t:.2f} {tau_t:.2f}'
        if len(m.group(2).strip().split()) >= 2
        else f'tau-t       = {tau_t:.2f}',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 添加注释标记 (如果还没有)
    if "auto-adjusted" not in content.lower():
        content = (
            f"; 自动调整的 NPT 参数 (由 04_npt.py 生成)\n"
            f"; dt={dt}  nsteps={nsteps}  tau_p={tau_p:.1f}  tau_t={tau_t:.2f}\n"
            + content
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ==================== 遍历报告 ====================

class TraverseReport:
    """NPT 遍历报告, 记录每次尝试的详细信息和质量分析。"""

    def __init__(self, input_dir: str, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.entries: List[dict] = []
        self.final_params: dict = {}
        self.generated_files: List[str] = []
        self.success = False

    def add_attempt(self, **kwargs) -> None:
        self.entries.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        })

    def generate(self) -> str:
        lines: List[str] = []
        sep = "=" * 80

        # ---- 头部 ----
        lines.append(sep)
        lines.append("GROMACS NPT 压力平衡 — 自动遍历报告")
        lines.append(sep)
        lines.append(f"  生成时间:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  完成时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  输入目录:     {self.input_dir}")
        lines.append(f"  输出目录:     {self.output_dir}")
        lines.append(f"  总共尝试:     {len(self.entries)} 次")
        lines.append(f"  最终结果:     {'[成功]' if self.success else '[失败]'}")
        lines.append("")

        # ---- 遍历策略 ----
        lines.append(sep)
        lines.append("参数遍历策略 (压力/温度不收敛时自动调整)")
        lines.append(sep)
        lines.append(f"  默认 MDP 模板: {_MDP_TEMPLATE_DIR}/{_NPT_MDP_TEMPLATE}")
        lines.append(f"  调整参数:      dt, nsteps, tau_p, tau_t")
        lines.append(f"  原始模板不会被修改, 每次尝试生成新的 mdp 文件")
        lines.append("")
        for i, (dt_val, nsteps_val, tau_p_val, tau_t_val, desc) in enumerate(
            _NPT_PARAM_TRAVERSE, 1
        ):
            lines.append(
                f"  [{i}] dt={dt_val}, nsteps={nsteps_val:>6d}, "
                f"tau_p={tau_p_val:.1f}, tau_t={tau_t_val:.2f}  — {desc}"
            )
        lines.append("")

        # ---- 遍历详情 ----
        lines.append(sep)
        lines.append("遍历详情")
        lines.append(sep)

        for i, entry in enumerate(self.entries, 1):
            status = "成功" if entry.get("status") == "success" else "失败"
            icon = "OK" if status == "成功" else "!!"
            lines.append(f"\n  [{icon}] 尝试 {i}  |  {entry['timestamp']}")
            lines.append(f"       参数:     dt={entry.get('dt', '?')}, "
                         f"nsteps={entry.get('nsteps', '?')}, "
                         f"tau_p={entry.get('tau_p', '?')}, "
                         f"tau_t={entry.get('tau_t', '?')}")
            lines.append(f"       结果:     {status}")

            quality = entry.get("quality", {})
            if quality:
                lines.append(f"       ---- NPT 质量检测 ----")
                lines.append(f"       模拟时长:   {quality.get('sim_time', '?')} ps")
                lines.append(f"       目标温度:   {quality.get('temp_target', '?')} K")
                lines.append(f"       平均温度:   {quality.get('temp_average', '?')} K")
                lines.append(f"       温度 RMSD:  {quality.get('temp_rmsd', '?')} K")
                lines.append(f"       温度漂移:   {quality.get('temp_drift', '?')} K")
                lines.append(f"       目标压力:   {quality.get('pres_target', '?')} bar")
                lines.append(f"       平均压力:   {quality.get('pres_average', '?')} bar")
                lines.append(f"       压力 RMSD:  {quality.get('pres_rmsd', '?')} bar")
                lines.append(f"       压力漂移:   {quality.get('pres_drift', '?')} bar")
                lines.append(f"       平均密度:   {quality.get('dens_average', '?')} kg/m³")
                lines.append(f"       密度漂移:   {quality.get('dens_drift_percent', '?')} %")
                lines.append(f"       平均势能:   {quality.get('pot_average', '?')}")
                lines.append(f"       势能漂移:   {quality.get('pot_drift_percent', '?')} %")
                if quality.get("lincs_warning"):
                    lines.append(f"       [致命] LINCS WARNING")
                if quality.get("nan_inf"):
                    lines.append(f"       [致命] nan/inf 值")
                if quality.get("step_size_error"):
                    lines.append(f"       [致命] step size too large")

            if entry.get("errors"):
                lines.append(f"       ---- 检测到的问题 ----")
                for err_msg in entry["errors"]:
                    lines.append(f"       ! {err_msg}")

            if entry.get("mdp_file"):
                lines.append(f"       MDP 文件: {entry['mdp_file']}")
            if entry.get("duration") is not None:
                lines.append(f"       耗时:     {entry['duration']:.1f}s")

        # ---- 最终参数 ----
        if self.success and self.final_params:
            lines.append(f"\n{sep}")
            lines.append("最终采用参数")
            lines.append(sep)
            for key, value in self.final_params.items():
                lines.append(f"  {key:<20s}: {value}")

        # ---- 文件清单 ----
        if self.generated_files:
            lines.append(f"\n{sep}")
            lines.append("生成文件清单")
            lines.append(sep)
            for f in self.generated_files:
                lines.append(f"  {f}")

        lines.append(f"\n{sep}")
        lines.append("报告结束")
        lines.append(sep)

        return "\n".join(lines)

    def save(self) -> Path:
        report_path = self.output_dir / "traverse_report.txt"
        report_path.write_text(self.generate(), encoding="utf-8")
        return report_path


# ==================== NPT 运行器 ====================

class NPTRunner:
    """GROMACS NPT 压力平衡运行器。

    读取 NVT 步骤的输出文件, 执行 grompp + mdrun,
    检测温度/压力/密度收敛质量, 不收敛时自动调整 dt/nsteps/tau_p/tau_t 重试。

    用法:
        runner = NPTRunner()
        result = runner.run()
    """

    def __init__(
        self,
        input_dir: str = "MD_result/nvt",
        output_dir: str = "MD_result/npt",
        gmx_cmd: str = "gmx",
        gpu: bool = True,
        dry_run: bool = False,
        topol_dir: Optional[str] = None,
    ) -> None:
        # ---- 路径 ----
        self.project_root = Path(__file__).parent.parent.resolve()
        _input = Path(input_dir)
        self.input_dir = _input.resolve() if _input.is_absolute() else (self.project_root / _input).resolve()
        _output = Path(output_dir)
        self.output_dir = _output.resolve() if _output.is_absolute() else (self.project_root / _output).resolve()
        self.adjusted_mdp_dir = self.output_dir / "adjusted_mdp"

        # ---- topol.top 路径 ----
        # 优先使用主控脚本传入的 topol_dir (单一真相源);
        # 未传入时兼容独立运行场景，从 input_dir 推算。
        if topol_dir:
            _td = Path(topol_dir)
            _td = _td.resolve() if _td.is_absolute() else (self.project_root / _td).resolve()
            self.topol_path: Path = _td / "topol.top"
        else:
            self.topol_path = self.input_dir.parent / "pdb2gmx" / "topol.top"

        # ---- 配置 ----
        self.gmx_cmd = gmx_cmd
        self.gpu = gpu
        self.dry_run = dry_run

        # ---- 自动检测 ----
        self.nvt_gro: Optional[Path] = None
        self.nvt_cpt: Optional[Path] = None

        # ---- 运行时状态 ----
        self._total_attempts: int = 0
        self.report = TraverseReport(str(self.input_dir), self.output_dir)

    # =====================================================================
    #  内部工具
    # =====================================================================

    def _count_attempt(self) -> int:
        self._total_attempts += 1
        if self._total_attempts > _MAX_TOTAL_ATTEMPTS:
            raise NPTError(f"已达全局最大尝试次数 ({_MAX_TOTAL_ATTEMPTS}), 终止运行")
        return self._total_attempts

    def run_gmx(
        self,
        *cmd: str,
        cwd: Optional[Path] = None,
        stdin_input: Optional[str] = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess:
        """执行 GROMACS 命令的统一入口。"""
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
                print(f"{Color.RED}{stderr_text.strip()[:500]}{Color.RESET}",
                      file=sys.stderr)
            raise NPTError(
                f"命令失败: {' '.join(full_cmd)}",
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e
        except FileNotFoundError:
            raise NPTError(
                f"找不到 GROMACS 命令 '{self.gmx_cmd}', "
                f"请确认已安装并激活 GROMACS 环境"
            ) from None

    @staticmethod
    def _extract_error_summary(stderr: str, stdout: str) -> str:
        combined = (stderr + "\n" + stdout).strip()
        if not combined:
            return "(无输出)"
        keywords = ["fatal", "error", "not found", "unknown", "did not match",
                     "cannot", "converged", "lincs", "nan", "inf", "step size"]
        for line in combined.splitlines():
            if any(kw in line.lower() for kw in keywords):
                return line.strip()[:200]
        return "\n".join(
            line.strip() for line in combined.splitlines()[-3:] if line.strip()
        )[:200]

    # =====================================================================
    #  输入验证
    # =====================================================================

    def _detect_inputs(self) -> None:
        """自动检测输入目录中的文件。"""
        # 验证拓扑文件 (在父级目录)
        if not self.topol_path.is_file():
            raise NPTError(
                f"拓扑文件不存在: {self.topol_path}\n"
                f"请先运行 01_pdb2gmx.py 完成准备步骤"
            )

        # 验证输入目录
        if not self.input_dir.is_dir():
            raise NPTError(
                f"输入目录不存在: {self.input_dir}\n"
                f"请先运行 03_nvt.py 完成 NVT 温度平衡"
            )

        # 自动查找 nvt.gro
        nvt_gro_path = self.input_dir / "nvt.gro"
        if nvt_gro_path.is_file():
            self.nvt_gro = nvt_gro_path
        else:
            gro_files = sorted(self.input_dir.glob("*.gro"))
            if gro_files:
                self.nvt_gro = gro_files[0]
                warn(f"未找到 nvt.gro, 将使用: {self.nvt_gro.name}")
            else:
                raise NPTError(
                    f"未找到 .gro 文件于 {self.input_dir}\n"
                    f"请确认 NVT 温度平衡步骤已完成"
                )

        # 自动查找 nvt.cpt (checkpoint 文件, 用于 continuation)
        nvt_cpt_path = self.input_dir / "nvt.cpt"
        if nvt_cpt_path.is_file():
            self.nvt_cpt = nvt_cpt_path
        else:
            # 尝试模糊匹配
            cpt_files = sorted(self.input_dir.glob("*.cpt"))
            if cpt_files:
                self.nvt_cpt = cpt_files[0]
                warn(f"未找到 nvt.cpt, 将使用: {self.nvt_cpt.name}")
            else:
                warn("未找到 .cpt checkpoint 文件, grompp 将不使用 -t 参数")
                warn("NPT 需要从 NVT 继续, 缺少 checkpoint 可能导致速度不连续")

        info(f"输入目录:       {self.input_dir}")
        info(f"拓扑文件:       {self.topol_path}")
        info(f"NVT 结构文件:   {self.nvt_gro}")
        if self.nvt_cpt:
            info(f"NVT 检查点:     {self.nvt_cpt}")

    # =====================================================================
    #  grompp + mdrun
    # =====================================================================

    def _run_grompp(self, mdp_path: Path, attempt_label: str) -> Path:
        """执行 grompp 生成 npt.tpr, -maxwarn 递增重试。

        NPT grompp 使用 -r 指定位置约束参考坐标, -t 指定 NVT checkpoint。

        Returns:
            生成的 npt.tpr 路径
        """
        tpr_path = self.output_dir / "npt.tpr"

        # 构建基础命令参数
        common_args = [
            "-f", str(mdp_path),
            "-c", str(self.nvt_gro),
            "-r", str(self.nvt_gro),
            "-p", str(self.topol_path),
            "-o", str(tpr_path),
        ]

        # 如果有 checkpoint 文件, 添加 -t 参数
        if self.nvt_cpt is not None:
            common_args.extend(["-t", str(self.nvt_cpt)])

        for maxwarn in _MAXWARN_SEQUENCE:
            self._count_attempt()
            if maxwarn > _MAXWARN_SEQUENCE[0]:
                info(f"  grompp 重试: -maxwarn {maxwarn}")

            try:
                self.run_gmx(
                    "grompp",
                    *common_args,
                    "-maxwarn", str(maxwarn),
                    cwd=self.output_dir,
                )
                info(f"  grompp 完成 → {tpr_path}")
                return tpr_path
            except NPTError as e:
                warn(f"  grompp 失败 (maxwarn={maxwarn})")
                if maxwarn == _MAXWARN_SEQUENCE[-1]:
                    raise NPTError(
                        f"grompp ({attempt_label}): 所有 -maxwarn 均已尝试完毕",
                        stderr=e.stderr, stdout=e.stdout,
                    ) from e

        raise NPTError(f"grompp ({attempt_label}): 未能生成 npt.tpr")

    def _run_mdrun(self) -> subprocess.CompletedProcess:
        """执行 mdrun, GPU 失败时自动降级为 CPU。"""
        # GPU 尝试
        if self.gpu:
            for retry in range(_MDRUN_GPU_RETRY):
                self._count_attempt()
                if retry > 0:
                    info(f"  GPU mdrun 重试 ({retry + 1}/{_MDRUN_GPU_RETRY})")
                try:
                    return self.run_gmx(
                        "mdrun", "-deffnm", "npt", "-v", "-nb", "gpu",
                        cwd=self.output_dir,
                    )
                except NPTError as e:
                    combined = (e.stderr + e.stdout).lower()
                    if any(kw in combined for kw in [
                        "lincs warning", "nan", "inf",
                        "step size too large",
                    ]):
                        raise
                    warn(f"  GPU mdrun 失败 (第 {retry + 1} 次)")
            warn("  GPU 全部失败, 降级为 CPU")

        # CPU 尝试
        for retry in range(_MDRUN_CPU_RETRY):
            self._count_attempt()
            if retry > 0:
                info(f"  CPU mdrun 重试 ({retry + 1}/{_MDRUN_CPU_RETRY})")
            try:
                return self.run_gmx(
                    "mdrun", "-deffnm", "npt", "-v",
                    cwd=self.output_dir,
                )
            except NPTError as e:
                warn(f"  CPU mdrun 失败 (第 {retry + 1} 次)")
                if retry == _MDRUN_CPU_RETRY - 1:
                    raise

        raise NPTError("mdrun: 所有重试均已用尽")

    # =====================================================================
    #  单次 NPT 尝试
    # =====================================================================

    def _try_npt_once(
        self, dt: float, nsteps: int, tau_p: float, tau_t: float,
        attempt_label: str,
    ) -> Tuple[bool, NPTQualityResult, Path]:
        """执行一次完整的 NPT (grompp + mdrun) 并检测质量。

        Args:
            dt:            时间步长 (ps)
            nsteps:        模拟步数
            tau_p:         压力耦合时间常数
            tau_t:         温度耦合时间常数
            attempt_label: 本次尝试的说明标签 (如 "attempt_1")

        Returns:
            (success, quality_result, mdp_path_used)
        """
        # ---- 创建调整后的 mdp 文件 ----
        template = _read_template_mdp(self.project_root)
        adjusted_mdp = self.adjusted_mdp_dir / f"npt_{attempt_label}.mdp"
        _create_adjusted_mdp(template, dt, nsteps, tau_p, tau_t, adjusted_mdp)
        info(f"  使用 MDP: {adjusted_mdp.name}  "
             f"(dt={dt}, nsteps={nsteps}, tau_p={tau_p:.1f}, tau_t={tau_t:.2f})")

        # ---- grompp ----
        self._run_grompp(adjusted_mdp, attempt_label)

        # ---- mdrun ----
        info(f"  运行 mdrun (NPT {'GPU' if self.gpu else 'CPU'}) ...")
        try:
            self._run_mdrun()
        except NPTError as e:
            # mdrun 崩溃 (如 LINCS error 导致直接退出)
            quality = NPTQualityResult()
            quality.errors.append(
                f"mdrun 异常退出: {e.stderr[:200] if e.stderr else str(e)}"
            )
            combined = (e.stderr + e.stdout).lower()
            if "lincs" in combined:
                quality.has_lincs_warning = True
            if "nan" in combined or "inf" in combined:
                quality.has_nan_inf = True
            if "step size" in combined:
                quality.has_step_size_error = True
            return False, quality, adjusted_mdp

        # ---- 分析 NPT 质量 ----
        log_path = self.output_dir / "npt.log"
        quality = _parse_npt_log(log_path)

        # 目标温度/压力必须来自实际使用的 mdp (绝不硬编码兜底)
        temp_target, pres_target = _read_mdp_targets(
            adjusted_mdp.read_text(encoding="utf-8")
        )
        if temp_target is None:
            raise NPTConfigError(
                f"无法从 {adjusted_mdp.name} 读取目标温度 (ref-t / gen-temp 均缺失),\n"
                f"请检查模板 {_MDP_TEMPLATE_DIR}/{_NPT_MDP_TEMPLATE} 是否包含 ref-t"
            )
        if pres_target is None:
            raise NPTConfigError(
                f"无法从 {adjusted_mdp.name} 读取目标压力 (ref-p 缺失),\n"
                f"请检查模板 {_MDP_TEMPLATE_DIR}/{_NPT_MDP_TEMPLATE} 是否包含 ref-p"
            )
        quality.temp_target = temp_target
        quality.pres_target = pres_target

        # 用我们设置的参数覆盖 (确保一致性)
        quality.dt = dt
        quality.total_steps = nsteps
        if quality.sim_time is None:
            quality.sim_time = dt * nsteps

        # ---- 从 .edr 文件提取温度/压力/密度/势能并分析平衡质量 ----
        # 这是 NPT 质量检测的核心步骤。使用 gmx energy (GROMACS 官方工具)
        # 从 .edr 文件提取热力学量时间序列, 确保跨 GROMACS 版本兼容。
        # (之前从 .log Statistics 块解析的方式已废弃 — 格式跨版本不稳定)
        edr_path = self.output_dir / "npt.edr"
        _analyze_edr_quality(
            edr_path, self.gmx_cmd, quality, self.dry_run,
            temp_target=temp_target, pres_target=pres_target,
        )

        # 综合判断 — equilibrated 已由 _analyze_edr_quality 综合判定
        # (包含致命错误 + 温度/压力/密度/势能质量)
        success = quality.equilibrated

        return success, quality, adjusted_mdp

    # =====================================================================
    #  主流程
    # =====================================================================

    def run(self) -> dict:
        """执行 NPT 压力平衡, 不收敛时自动调整 dt/nsteps/tau_p/tau_t 重试。

        Returns:
            dict:
                success          — bool
                dt               — float  最终采用的 dt
                nsteps           — int    最终采用的 nsteps
                tau_p            — float  最终采用的 tau_p
                tau_t            — float  最终采用的 tau_t
                temp_average     — float  最终平均温度 (K)
                temp_target      — float  目标温度 (K)
                pres_average     — float  最终平均压力 (bar)
                pres_target      — float  目标压力 (bar)
                dens_average     — float  最终平均密度 (kg/m³)
                output_dir       — str    输出目录
                report_path      — str    遍历报告路径
                total_attempts   — int    总尝试次数
                error            — str    (仅失败时)
        """
        banner("GROMACS NPT 压力平衡 — 自动控制脚本")

        try:
            # ---- 前置检查 ----
            self._detect_inputs()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adjusted_mdp_dir.mkdir(parents=True, exist_ok=True)
            info(f"输出目录: {self.output_dir}")

            # ---- 遍历 NPT 参数 ----
            tried_params: set = set()
            final_quality: Optional[NPTQualityResult] = None
            final_dt: float = 0.0
            final_nsteps: int = 0
            final_tau_p: float = 0.0
            final_tau_t: float = 0.0

            for dt, nsteps, tau_p, tau_t, desc in _NPT_PARAM_TRAVERSE:
                param_key = (dt, nsteps, tau_p, tau_t)
                if param_key in tried_params:
                    continue
                tried_params.add(param_key)

                attempt_idx = len(self.report.entries) + 1
                attempt_label = f"attempt_{attempt_idx}"

                info(f"\n{'─' * 50}")
                info(f"尝试 [{attempt_idx}]: {desc}  "
                     f"(dt={dt}, nsteps={nsteps}, tau_p={tau_p:.1f}, tau_t={tau_t:.2f})")

                t0 = time.time()
                try:
                    success, quality, mdp_path = self._try_npt_once(
                        dt, nsteps, tau_p, tau_t, attempt_label,
                    )
                except NPTConfigError:
                    # 配置/解析类错误 — 换参数无法解决, 立即终止阶梯
                    raise
                except NPTError as e:
                    dt_elapsed = time.time() - t0
                    warn(f"NPT 尝试失败 (命令执行错误): {e}")
                    self.report.add_attempt(
                        step="npt",
                        status="failed",
                        dt=dt,
                        nsteps=nsteps,
                        tau_p=tau_p,
                        tau_t=tau_t,
                        mdp_file=str(
                            self.adjusted_mdp_dir / f"npt_{attempt_label}.mdp"
                        ),
                        errors=[str(e)],
                        error_summary=self._extract_error_summary(
                            e.stderr, e.stdout
                        ),
                        duration=dt_elapsed,
                    )
                    continue

                dt_elapsed = time.time() - t0

                # 构建质量详情
                quality_detail = {
                    "sim_time": (
                        f"{quality.sim_time:.0f}"
                        if quality.sim_time else "?"
                    ),
                    "temp_target": (
                        f"{quality.temp_target:.2f}"
                        if quality.temp_target else "?"
                    ),
                    "temp_average": (
                        f"{quality.temp_average:.2f}"
                        if quality.temp_average else "?"
                    ),
                    "temp_rmsd": (
                        f"{quality.temp_rmsd:.2f}"
                        if quality.temp_rmsd else "?"
                    ),
                    "temp_drift": (
                        f"{quality.temp_drift:+.2f}"
                        if quality.temp_drift else "?"
                    ),
                    "pres_target": (
                        f"{quality.pres_target:.2f}"
                        if quality.pres_target else "?"
                    ),
                    "pres_average": (
                        f"{quality.pres_average:.2f}"
                        if quality.pres_average else "?"
                    ),
                    "pres_rmsd": (
                        f"{quality.pres_rmsd:.2f}"
                        if quality.pres_rmsd else "?"
                    ),
                    "pres_drift": (
                        f"{quality.pres_drift:+.2f}"
                        if quality.pres_drift else "?"
                    ),
                    "dens_average": (
                        f"{quality.dens_average:.2f}"
                        if quality.dens_average else "?"
                    ),
                    "dens_drift_percent": (
                        f"{quality.dens_drift_percent:+.1f}"
                        if quality.dens_drift_percent else "?"
                    ),
                    "pot_average": (
                        f"{quality.pot_average:.4e}"
                        if quality.pot_average else "?"
                    ),
                    "pot_drift_percent": (
                        f"{quality.pot_drift_percent:+.1f}"
                        if quality.pot_drift_percent else "?"
                    ),
                    "lincs_warning": quality.has_lincs_warning,
                    "nan_inf": quality.has_nan_inf,
                    "step_size_error": quality.has_step_size_error,
                }

                if success:
                    info(f"[成功] NPT 压力平衡完成 — {desc}")
                    info(f"  温度:  {quality_detail['temp_average']} K  "
                         f"(目标: {quality_detail['temp_target']} K, "
                         f"RMSD: {quality_detail['temp_rmsd']} K)")
                    info(f"  压力:  {quality_detail['pres_average']} bar  "
                         f"(目标: {quality_detail['pres_target']} bar, "
                         f"RMSD: {quality_detail['pres_rmsd']} bar)")
                    info(f"  密度:  {quality_detail['dens_average']} kg/m³")
                    info(f"  时长:  {quality_detail['sim_time']} ps")
                    info(f"  耗时:  {dt_elapsed:.1f}s")

                    self.report.add_attempt(
                        step="npt",
                        status="success",
                        dt=dt,
                        nsteps=nsteps,
                        tau_p=tau_p,
                        tau_t=tau_t,
                        quality=quality_detail,
                        mdp_file=str(mdp_path),
                        duration=dt_elapsed,
                    )

                    final_quality = quality
                    final_dt = dt
                    final_nsteps = nsteps
                    final_tau_p = tau_p
                    final_tau_t = tau_t
                    break
                else:
                    warn(f"[失败] NPT 未收敛 — {desc}")
                    if quality.errors:
                        for e_msg in quality.errors:
                            warn(f"  ! {e_msg}")
                    self.report.add_attempt(
                        step="npt",
                        status="failed",
                        dt=dt,
                        nsteps=nsteps,
                        tau_p=tau_p,
                        tau_t=tau_t,
                        quality=quality_detail,
                        errors=quality.errors,
                        mdp_file=str(mdp_path),
                        duration=dt_elapsed,
                    )
                    final_quality = quality
                    final_dt = dt
                    final_nsteps = nsteps
                    final_tau_p = tau_p
                    final_tau_t = tau_t

            # ---- 收尾 ----
            if final_quality is not None and final_quality.equilibrated:
                self.report.success = True
                self.report.final_params = {
                    "最终 dt": f"{final_dt}",
                    "最终 nsteps": final_nsteps,
                    "最终 tau_p": f"{final_tau_p:.1f}",
                    "最终 tau_t": f"{final_tau_t:.2f}",
                    "模拟时长 (ps)": (
                        f"{final_quality.sim_time:.0f}"
                        if final_quality.sim_time else "?"
                    ),
                    "平均温度 (K)": (
                        f"{final_quality.temp_average:.2f}"
                        if final_quality.temp_average else "?"
                    ),
                    "目标温度 (K)": (
                        f"{final_quality.temp_target:.2f}"
                        if final_quality.temp_target else "?"
                    ),
                    "温度 RMSD (K)": (
                        f"{final_quality.temp_rmsd:.2f}"
                        if final_quality.temp_rmsd else "?"
                    ),
                    "平均压力 (bar)": (
                        f"{final_quality.pres_average:.2f}"
                        if final_quality.pres_average else "?"
                    ),
                    "目标压力 (bar)": (
                        f"{final_quality.pres_target:.2f}"
                        if final_quality.pres_target else "?"
                    ),
                    "压力 RMSD (bar)": (
                        f"{final_quality.pres_rmsd:.2f}"
                        if final_quality.pres_rmsd else "?"
                    ),
                    "平均密度 (kg/m³)": (
                        f"{final_quality.dens_average:.2f}"
                        if final_quality.dens_average else "?"
                    ),
                    "总耗时": (
                        f"{(datetime.now() - self.report.start_time).total_seconds():.1f}s"
                    ),
                }
                self._collect_files()
                report_path = self.report.save()

                banner("NPT 完成 — 压力平衡成功")
                info(f"遍历报告: {report_path}")

                return {
                    "success": True,
                    "dt": final_dt,
                    "nsteps": final_nsteps,
                    "tau_p": final_tau_p,
                    "tau_t": final_tau_t,
                    "temp_average": final_quality.temp_average,
                    "temp_target": final_quality.temp_target,
                    "pres_average": final_quality.pres_average,
                    "pres_target": final_quality.pres_target,
                    "dens_average": final_quality.dens_average,
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }
            else:
                self.report.success = False
                self._collect_files()
                report_path = self.report.save()

                banner("NPT 失败 — 压力平衡未达标")
                error("所有参数组合均已尝试完毕, 压力仍未稳定")
                info(f"遍历报告: {report_path}")

                return {
                    "success": False,
                    "error": "NPT 失败: 所有 dt/nsteps/tau_p/tau_t 组合均已尝试",
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }

        except NPTError as e:
            self.report.success = False
            self.report.final_params["错误信息"] = str(e)
            self._collect_files()
            report_path = self.report.save()

            banner("NPT 失败")
            error(str(e))
            info(f"遍历报告: {report_path}")

            return {
                "success": False,
                "error": str(e),
                "output_dir": str(self.output_dir),
                "report_path": str(report_path),
                "total_attempts": self._total_attempts,
            }

        except KeyboardInterrupt:
            self.report.success = False
            self.report.final_params["错误信息"] = "用户中断 (Ctrl+C)"
            self._collect_files()
            report_path = self.report.save()

            warn("\n用户中断")
            info(f"遍历报告: {report_path}")

            return {
                "success": False,
                "error": "用户中断",
                "output_dir": str(self.output_dir),
                "report_path": str(report_path),
            }

    def _collect_files(self) -> None:
        """收集输出目录中的生成文件列表。"""
        collected: List[str] = []
        key_patterns = [
            "npt.tpr", "npt.gro", "npt.log", "npt.edr", "npt.trr",
            "npt.cpt", "npt.xtc", "mdout.mdp",
        ]
        for fname in key_patterns:
            fpath = self.output_dir / fname
            if fpath.is_file():
                size_kb = fpath.stat().st_size / 1024
                collected.append(f"{fname:<30s} {size_kb:>10.1f} KB")

        # adjusted_mdp 下的文件
        if self.adjusted_mdp_dir.is_dir():
            for mdp_file in sorted(self.adjusted_mdp_dir.glob("*.mdp")):
                rel = str(mdp_file.relative_to(self.output_dir))
                collected.append(f"{rel:<30s} {'(参数文件)':>10s}")

        self.report.generated_files = collected


# =========================================================================
#  公共 API — 供 run_md.py 通过 import 调用
# =========================================================================

def run_npt(
    input_dir: str = "MD_result/nvt",
    output_dir: str = "MD_result/npt",
    gmx_cmd: str = "gmx",
    gpu: bool = True,
    dry_run: bool = False,
    topol_dir: Optional[str] = None,
) -> dict:
    """执行 GROMACS NPT 压力平衡，自动检测温度/压力/密度收敛质量并调整参数。

    从 NVT 步骤的输出目录读取 nvt.gro + nvt.cpt，从 topol_dir 读取 topol.top，
    运行 grompp + mdrun，不收敛时自动调整 dt/nsteps/tau_p/tau_t 重试。

    Args:
        input_dir:  输入目录 (包含 nvt.gro + nvt.cpt), 默认 MD_result/nvt
        output_dir: 输出目录, 默认 MD_result/npt
        gmx_cmd:    GROMACS 命令, 默认 "gmx"
        gpu:        是否使用 GPU, 默认 True (失败自动降级 CPU)
        dry_run:    只打印命令不执行
        topol_dir:  拓扑文件目录 (包含 topol.top), 默认自动推算

    Returns:
        dict:
            success          — bool
            dt               — float  最终采用的 dt
            nsteps           — int    最终采用的 nsteps
            tau_p            — float  最终采用的 tau_p
            tau_t            — float  最终采用的 tau_t
            temp_average     — float  最终平均温度 (K)
            temp_target      — float  目标温度 (K)
            pres_average     — float  最终平均压力 (bar)
            pres_target      — float  目标压力 (bar)
            dens_average     — float  最终平均密度 (kg/m³)
            output_dir       — str    输出目录
            report_path      — str    traverse_report.txt 路径
            total_attempts   — int    总尝试次数
            error            — str    (仅失败时)

    Example:
        >>> import importlib
        >>> npt = importlib.import_module("04_npt")
        >>> result = npt.run_npt()
        >>> print(result["pres_average"])

    在 run_md.py 中集成示例:
        import importlib
        npt = importlib.import_module("04_npt")
        result = npt.run_npt(input_dir=str(self.result_dir / "nvt"),
                             topol_dir="MD_result/pdb2gmx")
        if not result["success"]:
            raise MDError(1, stderr=result.get("error", ""))
    """
    runner = NPTRunner(
        input_dir=input_dir,
        output_dir=output_dir,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
        topol_dir=topol_dir,
    )
    return runner.run()


# =========================================================================
#  CLI 入口 — 作为独立脚本运行
# =========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="04_npt.py",
        description="GROMACS NPT 压力平衡 — 自动检测温度/压力/密度收敛并调整参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 04_npt.py
  python 04_npt.py --input MD_result/nvt --output MD_result/npt
  python 04_npt.py --dry-run
  python 04_npt.py --no-gpu
        """,
    )
    parser.add_argument(
        "--input", default=DEFAULTS["input_dir"],
        help=f"输入目录 (包含 nvt.gro + nvt.cpt, 默认: {DEFAULTS['input_dir']})",
    )
    parser.add_argument(
        "--output", default=DEFAULTS["output_dir"],
        help=f"输出目录 (默认: {DEFAULTS['output_dir']})",
    )
    parser.add_argument(
        "--gmx", default=DEFAULTS["gmx"],
        help=f"GROMACS 命令 (默认: {DEFAULTS['gmx']})",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, dest="gpu",
        help="使用 GPU 加速 (默认)",
    )
    parser.add_argument(
        "--no-gpu", action="store_false", default=True, dest="gpu",
        help="仅使用 CPU",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="只打印命令不执行",
    )

    args = parser.parse_args(argv)

    result = run_npt(
        input_dir=args.input,
        output_dir=args.output,
        gmx_cmd=args.gmx,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )

    if result["success"]:
        print(f"\n{'=' * 60}")
        print(f"  NPT 完成 — 压力平衡成功")
        print(f"  温度:  {result['temp_average']} K (目标: {result['temp_target']} K)")
        print(f"  压力:  {result['pres_average']} bar (目标: {result['pres_target']} bar)")
        print(f"  密度:  {result['dens_average']} kg/m³")
        print(f"  报告:  {result['report_path']}")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"  NPT 失败")
        print(f"  原因:  {result.get('error', '未知')}")
        print(f"  报告:  {result.get('report_path', 'N/A')}")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
