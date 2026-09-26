#!/usr/bin/env python3
"""
GROMACS 成品模拟 (Production MD) 自动化脚本
MD 流程的最后一步: 恒温恒压成品采样, 自动检测轨迹稳定性, 崩溃时自动调整参数重试

检测策略:
  - 解析 mdrun 输出, 检测温度/压力/密度/势能是否稳定 (无明显漂移)
  - 通过 gmx energy 提取温度/压力/势能/密度时间序列进行定量分析
  - 检测致命错误: LINCS WARNING / nan / inf / step size too large
  - 成品 MD 无"收敛目标", 以轨迹质量 (稳定性) 为成功标准

不收敛/崩溃时自动调整参数 (在原 mdp 文件外创建新文件, 原始 md.mdp 不被修改):
  dt:     0.002 → 0.001 → 0.0005
  nsteps: 500000 (1ns) → 1000000 (2ns) → 2000000 (4ns)
  tau_p:  2.0 → 1.0 → 0.5
  tau_t:  0.1 → 0.05

输入:  MD_result/npt/       (从 NPT 步骤读取 npt.gro + npt.cpt, 从上级目录读取 topol.top)
输出:  MD_result/md/
  ├── md.tpr / md.gro / md.log / md.edr / md.trr / md.cpt / md.xtc
  ├── adjusted_mdp/           # 调整后的参数文件 (原始 md.mdp 不被修改)
  │   ├── md_attempt_1.mdp
  │   └── ...
  └── traverse_report.txt     # 遍历报告

用法:
  # 独立脚本
  python 05_md.py

  # 被 run_md.py 导入
  import importlib
  md = importlib.import_module("05_md")
  result = md.run_md()
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
    "input_dir": "MD_result/npt",
    "output_dir": "MD_result/md",
    "gmx": "gmx",
    "dry_run": False,
}

# MDP 模板路径 (项目根目录下)
_MDP_TEMPLATE_DIR = "mdp_files"
_MD_MDP_TEMPLATE = "md.mdp"

# grompp -maxwarn 递增序列
_MAXWARN_SEQUENCE: List[int] = [10, 20, 50, 100]

# GPU → CPU 降级重试
_MDRUN_GPU_RETRY = 3
_MDRUN_CPU_RETRY = 3

# 全局最大尝试次数
_MAX_TOTAL_ATTEMPTS = 100


# ==================== MD 参数遍历策略 ====================

# 不收敛/崩溃时的 dt + nsteps + tau_p + tau_t 组合遍历顺序
# 原则: 先收紧耦合, 再减小步长, 最后延长模拟时间确保充分采样
_MD_PARAM_TRAVERSE: List[Tuple[float, int, float, float, str]] = [
    # (dt, nsteps, tau_p, tau_t, 说明)
    (0.002, 500000,  2.0, 0.1,  "默认参数 (1 ns)"),
    (0.002, 500000,  1.0, 0.1,  "收紧压力耦合"),
    (0.002, 500000,  1.0, 0.05, "收紧温压耦合"),
    (0.001, 500000,  1.0, 0.05, "减半步长 + 收紧耦合"),
    (0.001, 1000000, 1.0, 0.05, "减半步长 + 延长至 2 ns"),
    (0.0005, 1000000, 0.5, 0.05,"更小步长 + 2 ns + 紧耦合"),
    (0.0005, 2000000, 0.5, 0.05,"最严格: 最小步长 + 4 ns + 紧耦合"),
]


# ==================== 异常类 ====================

class MDError(Exception):
    """MD 步骤中的错误。"""

    def __init__(self, message: str, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


class MDConfigError(MDError):
    """配置/解析类错误 — 重试阶梯 (dt/nsteps/tau_p/tau_t) 无法改善, 应立即停止。"""


# ==================== MD 质量检测 ====================

class MDQualityResult:
    """成品 MD 轨迹质量检测结果。

    成品 MD 无收敛目标, 以轨迹稳定性为核心指标。
    """

    def __init__(self) -> None:
        self.stable: bool = False
        # 完成状态
        self.completed: bool = False                        # 模拟是否正常完成
        self.completed_steps: int = 0                       # 实际完成的步数
        # 温度
        self.temp_average: Optional[float] = None           # 平均温度 (K)
        self.temp_target: Optional[float] = None             # 目标温度 (K)
        self.temp_rmsd: Optional[float] = None              # 温度波动 RMSD (K)
        self.temp_drift: Optional[float] = None             # 温度漂移 (K)
        # 压力
        self.pres_average: Optional[float] = None           # 平均压力 (bar)
        self.pres_target: Optional[float] = None            # 目标压力 (bar)
        self.pres_rmsd: Optional[float] = None              # 压力波动 RMSD (bar)
        self.pres_drift: Optional[float] = None             # 压力漂移 (bar)
        # 密度
        self.dens_average: Optional[float] = None           # 平均密度 (kg/m³)
        self.dens_drift_percent: Optional[float] = None     # 密度漂移百分比
        # 势能
        self.pot_average: Optional[float] = None            # 平均势能
        self.pot_initial: Optional[float] = None            # 初始势能 (前10%平均)
        self.pot_final: Optional[float] = None              # 最终势能 (后10%平均)
        self.pot_drift_percent: Optional[float] = None      # 势能漂移百分比
        # 时间序列
        self.temp_values: List[float] = []                  # 温度时间序列
        self.pres_values: List[float] = []                  # 压力时间序列
        self.pot_values: List[float] = []                   # 势能时间序列
        self.dens_values: List[float] = []                  # 密度时间序列
        # 模拟信息
        self.total_steps: int = 0
        self.dt: Optional[float] = None                     # 时间步长 (ps)
        self.sim_time: Optional[float] = None               # 模拟时长 (ps)
        # 致命错误
        self.has_lincs_warning: bool = False
        self.has_nan_inf: bool = False
        self.has_step_size_error: bool = False
        # 额外质量指标
        self.traj_file_exists: bool = False                 # .xtc 轨迹文件是否存在
        self.traj_file_size_mb: Optional[float] = None      # .xtc 文件大小 (MB)
        # 错误收集
        self.errors: List[str] = []


# ==================== 日志解析 ====================

_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _stats_block_mean(stats_lines: List[str], label: str) -> Optional[float]:
    """从 GROMACS 统计块 (Statistics over N steps) 中取出 label 对应的均值。

    已知两种排版:

      旧版 —— 标签与数值同一行:
          Temperature        300.123        0.012        1.234     0.567
      新版 —— 2026.2, 标签行 + 数值行, 5 个标签一组:
          Potential    Kinetic En.   Total Energy  Conserved En.    Temperature
        -8.24333e+05    1.42195e+05   -6.82138e+05   -6.81986e+05    2.99987e+02

    新版里 Temperature 往往是该行的**最后一个**标签, 它的值在下一行的**第 5 列**,
    数值与标签按结束列右对齐。原先用 'Temperature 后面跟三个数' 去抓, 会抓到下一行
    **开头**的三个数 —— 也就是势能/动能/总能量, 于是得到 "平均温度 -824333 K"
    (即历史上一度出现的 -824711 K), 触发温度硬校验失败、判轨迹不稳定并反复重跑。
    这里按结束列配对, 与标签的结束列最接近的那个数值即为该标签的值。
    """
    for i, line in enumerate(stats_lines):
        m = re.search(rf'\b{re.escape(label)}\b', line, re.IGNORECASE)
        if not m:
            continue

        # 旧版: 同一行标签后面还有数字, 第一个即为均值。
        same_line = [n for n in _NUM_RE.finditer(line) if n.start() >= m.end()]
        if same_line:
            return float(same_line[0].group())

        # 新版: 数值在下一行, 且该行应为纯数值行。
        if i + 1 >= len(stats_lines):
            continue
        nxt = stats_lines[i + 1]
        if _NUM_RE.sub('', nxt).strip() or not _NUM_RE.search(nxt):
            continue
        cands = list(_NUM_RE.finditer(nxt))
        return float(min(cands, key=lambda n: abs(n.end() - m.end())).group())

    return None


def _parse_md_log(
    log_path: Path,
    expected_nsteps: int,
    temp_target: Optional[float] = None,
    pres_target: Optional[float] = None,
) -> MDQualityResult:
    """解析 GROMACS MD 的 .log 文件, 检测轨迹质量。

    成品 MD 无收敛目标, 检查项目:
      1. 致命错误: LINCS WARNING / nan / inf / step size too large
      2. 完成状态: 是否跑完所有 nsteps
      3. 温度稳定性: 平均值合理性、RMSD、漂移
      4. 压力稳定性: 平均值合理性、RMSD、漂移
      5. 密度稳定性: 是否收敛无漂移
      6. 势能稳定性: 是否有明显漂移
      7. 模拟参数: dt, nsteps, ref_t, ref_p

    Args:
        log_path:         md.log 文件路径
        expected_nsteps: 预期的总步数 (用于判断是否完成)
        temp_target:     目标温度 (K) — 由调用方从实际使用的 mdp 中读取并传入,
                         优先于日志解析结果
        pres_target:     目标压力 (bar) — 同上

    Returns:
        MDQualityResult 包含所有检测结果
    """
    result = MDQualityResult()

    # 目标值以调用方传入的实际 mdp 为准 (日志解析仅作兜底)
    if temp_target is not None:
        result.temp_target = temp_target
    if pres_target is not None:
        result.pres_target = pres_target

    if not log_path.is_file():
        result.errors.append("日志文件不存在")
        return result

    content = log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    # ---- 1. 致命错误检测 ----

    if re.search(r'LINCS\s+WARNING', content, re.IGNORECASE):
        result.has_lincs_warning = True
        result.errors.append("LINCS WARNING — 约束算法失败, 体系中存在严重冲突")

    # nan/inf 检测: 只在模拟输出段搜索
    # (参数段含 "epsilon-rf = inf" 等合法值, 不能盲目全文搜索)
    _step_start = 0
    for i, line in enumerate(lines):
        if re.search(r'^\s*Step\s{2,}', line):
            _step_start = i
            break
    _step_content = "\n".join(lines[_step_start:]) if _step_start > 0 else content

    if re.search(r'\bnan\b', _step_content, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("检测到 nan 值 — 体系中存在非物理接触或参数错误")
    if re.search(r'\binf\b', _step_content, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("检测到 inf 值 — 体系中存在非物理接触或参数错误")

    if re.search(r'step\s+size\s+too\s+large', content, re.IGNORECASE):
        result.has_step_size_error = True
        result.errors.append("Step size too large — 力场梯度过大, 体系不稳定")

    # ---- 2. 完成状态检测 ----

    # Performance 块中有实际完成的步数
    # 格式: "              500000        0.624      100.0"
    # 或包含 "Performance" 行后跟步数统计
    perf_pattern = re.compile(
        r'Performance',
        re.IGNORECASE,
    )
    perf_lines_start = -1
    for i, line in enumerate(lines):
        if perf_pattern.search(line):
            perf_lines_start = i
            break

    if perf_lines_start >= 0:
        # 查找后面的数字行 (通常格式: step  time  performance%)
        for j in range(perf_lines_start + 1, min(perf_lines_start + 10, len(lines))):
            step_line = lines[j].strip()
            if step_line and not step_line.startswith(";") and not step_line.startswith("#"):
                parts = step_line.split()
                if len(parts) >= 1:
                    try:
                        result.completed_steps = int(parts[0])
                    except ValueError:
                        continue
                break

    # 备选: 从日志末尾 "Finished mdrun" 或 "Writing final coordinates" 判断
    if re.search(r'Finished\s+mdrun', content, re.IGNORECASE):
        result.completed = True
    elif re.search(r'Writing\s+final\s+coordinates', content, re.IGNORECASE):
        result.completed = True
    elif result.completed_steps >= expected_nsteps:
        result.completed = True
    elif result.completed_steps > 0:
        # 部分完成
        result.completed = False
        result.errors.append(
            f"模拟未完成: 仅跑 {result.completed_steps}/{expected_nsteps} 步"
        )

    # ---- 3. 提取模拟参数 ----

    # dt
    dt_match = re.search(r'(?:^|\n)\s*dt\s*=\s*([\d.]+)', content, re.IGNORECASE)
    if dt_match:
        result.dt = float(dt_match.group(1))

    # nsteps
    nsteps_match = re.search(r'(?:^|\n)\s*nsteps\s*=\s*(\d+)', content, re.IGNORECASE)
    if nsteps_match:
        result.total_steps = int(nsteps_match.group(1))

    # ref_t
    ref_t_match = re.search(
        r'(?:^|\n)\s*ref[-_]?t\s*[:=]\s*([\d.]+(?:\s+[\d.]+)*)',
        content, re.IGNORECASE,
    )
    if ref_t_match:
        ref_t_str = ref_t_match.group(1).strip()
        result.temp_target = float(ref_t_str.split()[0])

    # ref_p
    ref_p_match = re.search(
        r'(?:^|\n)\s*ref[-_]?p\s*[:=]\s*([\d.]+(?:\s+[\d.]+)*)',
        content, re.IGNORECASE,
    )
    if ref_p_match:
        ref_p_str = ref_p_match.group(1).strip()
        result.pres_target = float(ref_p_str.split()[0])

    # ---- 4. 从日志中解析每帧温度/压力/密度 ----
    # 成品 MD 的列式输出格式:
    #    Step           Time         Temperature    Pressure       Density
    #       0        0.0000          300.00000        1.00000     1000.000

    temp_vals: List[float] = []
    pres_vals: List[float] = []
    dens_vals: List[float] = []

    col_header_pattern = re.compile(
        r'Step\s+\S+\s+.*Temperature',
        re.IGNORECASE,
    )
    in_table = False
    col_indices: Optional[Tuple[int, int, int]] = None  # (temp_idx, pres_idx, dens_idx)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                in_table = False
            continue

        # 检测表头, 确定列索引
        if not in_table and col_header_pattern.search(stripped):
            in_table = True
            # "Step  Time  ...  Temperature  Pressure  Density  ..."
            headers = stripped.split()
            temp_idx, pres_idx, dens_idx = -1, -1, -1
            for hi, h in enumerate(headers):
                hl = h.lower()
                if hl.startswith("temp"):
                    temp_idx = hi
                elif hl.startswith("pres"):
                    pres_idx = hi
                elif hl.startswith("den") or hl == "density":
                    dens_idx = hi
            if temp_idx >= 0:
                col_indices = (temp_idx, pres_idx if pres_idx >= 0 else -1,
                               dens_idx if dens_idx >= 0 else -1)
            continue

        if in_table and col_indices:
            parts = stripped.split()
            ti, pi, di = col_indices
            try:
                if ti < len(parts):
                    val = float(parts[ti])
                    if 250.0 < val < 500.0:
                        temp_vals.append(val)
                if pi >= 0 and pi < len(parts):
                    val = float(parts[pi])
                    if -500.0 < val < 5000.0:
                        pres_vals.append(val)
                if di >= 0 and di < len(parts):
                    val = float(parts[di])
                    if 500.0 < val < 2000.0:
                        dens_vals.append(val)
            except (ValueError, IndexError):
                pass

    result.temp_values = temp_vals
    result.pres_values = pres_vals
    result.dens_values = dens_vals

    # ---- 5. 最终统计信息 ----
    # Statistics 块 (新版格式):
    #           Average       Err.Est.       RMSD  Tot. Drift
    #   Temperature    300.123        0.012        1.234     0.567
    #   Pressure         1.234        0.123       50.123    10.456
    #   Density       1000.123        0.456        1.234   -0.789

    stats_block_pattern = re.compile(
        r'Statistics\s+over\s+\d+\s+steps',
        re.IGNORECASE,
    )

    stats_start = -1
    for i, line in enumerate(lines):
        if stats_block_pattern.search(line):
            stats_start = i
            break

    if stats_start >= 0:
        stats_lines = lines[stats_start:min(stats_start + 80, len(lines))]
        stats_text = "\n".join(stats_lines)

        # 温度均值: 按结束列配对取值 (见 _stats_block_mean 的说明)。
        _temp_avg = _stats_block_mean(stats_lines, "Temperature")
        # 物理合理性守门: 统计块一旦解析错列就会给出势能量级的数值。此时留空即可,
        # 由 _try_md_once 中的 .edr 时序回填真实均值 —— 不能在此记 error,
        # 否则会把本可正常判定的轨迹误判为不稳定 (这正是旧 bug 的后果)。
        if _temp_avg is not None and 150.0 <= _temp_avg <= 500.0:
            result.temp_average = _temp_avg

        # 压力统计
        pres_stats_pattern = re.compile(
            r'Pressure\s+([\d.\-]+(?:[eE][+\-]?\d+)?)\s+'
            r'([\d.\-]+(?:[eE][+\-]?\d+)?)\s+'
            r'([\d.\-]+(?:[eE][+\-]?\d+)?)',
            re.IGNORECASE,
        )
        pres_match = pres_stats_pattern.search(stats_text)
        if pres_match:
            try:
                result.pres_average = float(pres_match.group(1))
                result.pres_rmsd = float(pres_match.group(3))
            except ValueError:
                pass

        # 密度统计
        dens_stats_pattern = re.compile(
            r'Density\s+([\d.\-]+(?:[eE][+\-]?\d+)?)\s+'
            r'([\d.\-]+(?:[eE][+\-]?\d+)?)\s+'
            r'([\d.\-]+(?:[eE][+\-]?\d+)?)',
            re.IGNORECASE,
        )
        dens_match = dens_stats_pattern.search(stats_text)
        if dens_match:
            try:
                result.dens_average = float(dens_match.group(1))
            except ValueError:
                pass

        # 势能统计
        pot_match = re.search(
            r'Potential\s+.*?([\d.\-]+(?:[eE][+\-]?\d+)?)\s+'
            r'([\d.\-]+(?:[eE][+\-]?\d+)?)',
            stats_text, re.IGNORECASE,
        )
        if pot_match:
            try:
                result.pot_average = float(pot_match.group(1))
            except ValueError:
                pass
        else:
            pot_simple_match = re.search(
                r'Potential\s+([\d.\-]+(?:[eE][+\-]?\d+)?)',
                stats_text, re.IGNORECASE,
            )
            if pot_simple_match:
                try:
                    result.pot_average = float(pot_simple_match.group(1))
                except ValueError:
                    pass

        # ---- 旧版格式兼容 ----
        if result.temp_average is None:
            old_temp_match = re.search(
                r'Temperature\s+([\d.\-]+(?:[eE][+\-]?\d+)?)',
                stats_text, re.IGNORECASE,
            )
            if old_temp_match:
                try:
                    result.temp_average = float(old_temp_match.group(1))
                except ValueError:
                    pass

    # ---- 6. 计算模拟时长 ----
    if result.dt is not None and result.total_steps > 0:
        result.sim_time = result.dt * result.total_steps

    # ---- 7. 温度稳定性分析 ----
    if temp_vals:
        n_temp = len(temp_vals)
        if n_temp >= 20:
            front_n = max(1, n_temp // 5)
            front_avg = sum(temp_vals[:front_n]) / front_n
            back_avg = sum(temp_vals[-front_n:]) / front_n
            result.temp_drift = back_avg - front_avg

    # ---- 8. 压力稳定性分析 ----
    if pres_vals:
        n_pres = len(pres_vals)
        if n_pres >= 20:
            front_n = max(1, n_pres // 5)
            front_avg = sum(pres_vals[:front_n]) / front_n
            back_avg = sum(pres_vals[-front_n:]) / front_n
            result.pres_drift = back_avg - front_avg

    # ---- 9. 密度稳定性分析 ----
    if dens_vals:
        n_dens = len(dens_vals)
        if n_dens >= 20:
            result.dens_average = sum(dens_vals) / n_dens
            front_n = max(1, n_dens // 5)
            front_avg = sum(dens_vals[:front_n]) / front_n
            back_avg = sum(dens_vals[-front_n:]) / front_n
            if abs(front_avg) > 1e-10:
                result.dens_drift_percent = (back_avg - front_avg) / abs(front_avg) * 100

    # ---- 9b. 平均值的稳健回填 ----
    # 修复外部审计发现的"误读": 统计块正则可能把势能/总能量等误读成温度 (如曾出现
    # -824711 K) / 压力缺失 (None)。逐帧值已在 250-500 K、-500-5000 bar 物理范围内过滤,
    # 更可靠, 因此优先采用逐帧平均值; 仅当逐帧缺失且统计块数值落在常温常压区间时才用统计块。
    if temp_vals:
        _avg_temp = sum(temp_vals) / len(temp_vals)
        if result.temp_average is None or not (150.0 <= result.temp_average <= 500.0):
            result.temp_average = _avg_temp
    if pres_vals:
        _avg_pres = sum(pres_vals) / len(pres_vals)
        if result.pres_average is None or not (-500.0 <= result.pres_average <= 5000.0):
            result.pres_average = _avg_pres

    # ---- 10. 综合判断 ----

    # 致命错误 → 直接失败
    if result.has_lincs_warning or result.has_nan_inf or result.has_step_size_error:
        result.stable = False
        return result

    # 未完成 → 不检查稳定性 (由崩溃处理逻辑接管)
    if not result.completed:
        if not result.errors:
            result.errors.append("模拟未正常完成")
        result.stable = False
        return result

    # ---- 温度稳定性 ----
    # 物理合理性硬校验: 与 temp_target 是否解析成功无关, 防止把不可能的温度当有效 (审计 P0)
    if result.temp_average is not None and not (100.0 <= result.temp_average <= 600.0):
        result.errors.append(
            f"平均温度不可信 (超出物理合理区间): {result.temp_average:.1f}K"
        )
    if result.temp_target is None:
        # 目标温度必须由调用方 (mdp) 或日志提供 — 缺失时不得静默通过
        result.errors.append(
            "无法确定目标温度: mdp 中缺少 ref-t / gen-temp, 日志中也未解析到"
        )
    if result.temp_average is not None and result.temp_target is not None:
        temp_diff = abs(result.temp_average - result.temp_target)
        if temp_diff > 20.0:
            result.errors.append(
                f"温度偏差过大: 平均={result.temp_average:.1f}K, "
                f"目标={result.temp_target:.1f}K, 偏差={temp_diff:.1f}K"
            )
        if result.temp_rmsd is not None and result.temp_rmsd > 20.0:
            result.errors.append(
                f"温度波动过大: RMSD={result.temp_rmsd:.1f}K (阈值 20.0K)"
            )

    if result.temp_drift is not None and abs(result.temp_drift) > 40.0:
        result.errors.append(
            f"温度漂移过大: {result.temp_drift:+.1f}K (前段→后段差异 > 40K)"
        )

    # ---- 压力稳定性 ----
    # 物理合理性硬校验: 压力缺失或非数值时判为不可信 (审计 P0)
    if result.pres_average is not None and not (-1000.0 <= result.pres_average <= 10000.0):
        result.errors.append(
            f"平均压力不可信 (超出物理合理区间): {result.pres_average:.1f}bar"
        )
    if result.pres_target is None:
        # 目标压力必须由调用方 (mdp) 或日志提供 — 缺失时不得静默通过
        result.errors.append(
            "无法确定目标压力: mdp 中缺少 ref-p, 日志中也未解析到"
        )
    if result.pres_average is not None and result.pres_target is not None:
        pres_diff = abs(result.pres_average - result.pres_target)
        if pres_diff > 100.0:
            result.errors.append(
                f"压力偏差过大: 平均={result.pres_average:.1f}bar, "
                f"目标={result.pres_target:.1f}bar, 偏差={pres_diff:.1f}bar"
            )
        if result.pres_rmsd is not None and result.pres_rmsd > 200.0:
            result.errors.append(
                f"压力波动过大: RMSD={result.pres_rmsd:.1f}bar (阈值 200.0bar)"
            )

    if result.pres_drift is not None and abs(result.pres_drift) > 200.0:
        result.errors.append(
            f"压力漂移过大: {result.pres_drift:+.1f}bar (前段→后段差异 > 200bar)"
        )

    # ---- 密度稳定性 ----
    if result.dens_drift_percent is not None and abs(result.dens_drift_percent) > 10.0:
        result.errors.append(
            f"密度漂移过大: {result.dens_drift_percent:+.1f}% (前后 20% 段差异 > 10%)"
        )

    # ---- 势能稳定性 ----
    if result.pot_values:
        n_pot = len(result.pot_values)
        if n_pot >= 20:
            front_n = max(1, n_pot // 5)
            result.pot_initial = sum(result.pot_values[:front_n]) / front_n
            result.pot_final = sum(result.pot_values[-front_n:]) / front_n
            if result.pot_initial and abs(result.pot_initial) > 1e-10:
                drift = (result.pot_final - result.pot_initial) / abs(result.pot_initial) * 100
                result.pot_drift_percent = drift
                if abs(drift) > 20.0:
                    result.errors.append(
                        f"势能漂移过大: {drift:+.1f}% (前后 20% 段差异 > 20%)"
                    )

    # ---- 数据完整性 ----
    if result.temp_average is None and not temp_vals:
        result.errors.append("无法从日志中提取温度信息, 请检查模拟是否正常完成")
    if result.pres_average is None and not pres_vals:
        # 压力数据缺失不一定致命 (旧版 GROMACS 可能不输出), 只警告
        pass

    # ---- 11. 最终判定 ----
    result.stable = (
        not result.has_lincs_warning
        and not result.has_nan_inf
        and not result.has_step_size_error
        and result.completed
        and len(result.errors) == 0
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

    try:
        run_gmx(
            [gmx_cmd, "energy", "-f", str(edr_path), "-o", "temp_analysis.xvg"],
            input=f"{term}\n0\n",
            capture_output=True,
            cwd=str(edr_path.parent),
            check=True,
        )

        xvg_path = edr_path.parent / "temp_analysis.xvg"
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
    """读取原始 md.mdp 模板内容。"""
    template_path = project_root / _MDP_TEMPLATE_DIR / _MD_MDP_TEMPLATE
    if not template_path.is_file():
        raise MDConfigError(
            f"MD 参数模板文件不存在: {template_path}\n"
            f"请确认 {_MDP_TEMPLATE_DIR}/{_MD_MDP_TEMPLATE} 位于项目根目录"
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

    调整 dt, nsteps, tau_p, tau_t 四个参数,
    同时自动协调 nstxout-compressed 等输出频率以匹配 nsteps。

    Args:
        template_content: 原始 md.mdp 的文本内容
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
        f'dt          = {dt}',
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

    # 调整 nstxout-compressed 使总帧数控制在 ~100 帧左右 (不过于密集)
    # 原模板 nstxout-compressed = 5000, 对应 500000/5000 = 100 帧
    target_frames = 100
    new_nstxout = max(1000, nsteps // target_frames)
    # 取整到最近的 500 倍数
    new_nstxout = ((new_nstxout + 250) // 500) * 500
    content = re.sub(
        r'^(\s*nstxout-compressed\s*=\s*)(.*)$',
        f'nstxout-compressed = {new_nstxout}',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 其他输出频率也按比例调整 (保持约 100 帧)
    new_nstout = max(1000, nsteps // target_frames)
    new_nstout = ((new_nstout + 250) // 500) * 500
    for param in ["nstvout", "nstenergy", "nstlog"]:
        content = re.sub(
            rf'^(\s*{param}\s*=\s*)(.*)$',
            f'{param:<15s} = {new_nstout}',
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # 添加注释标记 (如果还没有)
    if "auto-adjusted" not in content.lower():
        content = (
            f"; 自动调整的 MD 参数 (由 05_md.py 生成)\n"
            f"; dt={dt}  nsteps={nsteps}  tau_p={tau_p:.1f}  tau_t={tau_t:.2f}\n"
            + content
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ==================== 遍历报告 ====================

class TraverseReport:
    """MD 遍历报告, 记录每次尝试的详细信息和稳定性分析。"""

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
        lines.append("GROMACS 成品模拟 (Production MD) — 自动遍历报告")
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
        lines.append("参数遍历策略 (轨迹不稳定时自动调整)")
        lines.append(sep)
        lines.append(f"  默认 MDP 模板: {_MDP_TEMPLATE_DIR}/{_MD_MDP_TEMPLATE}")
        lines.append(f"  调整参数:      dt, nsteps, tau_p, tau_t")
        lines.append(f"  原始模板不会被修改, 每次尝试生成新的 mdp 文件")
        lines.append("")
        for i, (dt_val, nsteps_val, tau_p_val, tau_t_val, desc) in enumerate(
            _MD_PARAM_TRAVERSE, 1
        ):
            sim_ns = dt_val * nsteps_val / 1000.0
            lines.append(
                f"  [{i}] dt={dt_val}, nsteps={nsteps_val:>7d} ({sim_ns:.0f} ns), "
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
                lines.append(f"       ---- MD 质量检测 ----")
                lines.append(f"       完成状态:   {quality.get('completed', '?')}")
                lines.append(f"       完成步数:   {quality.get('completed_steps', '?')}")
                lines.append(f"       模拟时长:   {quality.get('sim_time', '?')} ps "
                             f"({quality.get('sim_time_ns', '?')} ns)")
                lines.append(f"       轨迹文件:   {quality.get('traj_file', '?')} "
                             f"({quality.get('traj_size_mb', '?')} MB)")
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
                mins = entry["duration"] / 60.0
                lines.append(f"       耗时:     {mins:.1f} min ({entry['duration']:.1f}s)")

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


# ==================== MD 运行器 ====================

class MDRunner:
    """GROMACS 成品模拟 (Production MD) 运行器。

    读取 NPT 步骤的输出文件, 执行 grompp + mdrun,
    检测轨迹稳定性, 不稳定时自动调整 dt/nsteps/tau_p/tau_t 重试。

    与 NPT 的关键区别:
      - grompp 不使用 -r 位置约束 (成品模拟不需要)
      - 模拟时长更长 (500000 步起, 可遍历到 2000000 步)
      - 成功标准是轨迹稳定性而非收敛

    用法:
        runner = MDRunner()
        result = runner.run()
    """

    def __init__(
        self,
        input_dir: str = "MD_result/npt",
        output_dir: str = "MD_result/md",
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
        self.npt_gro: Optional[Path] = None
        self.npt_cpt: Optional[Path] = None

        # ---- 运行时状态 ----
        self._total_attempts: int = 0
        self.report = TraverseReport(str(self.input_dir), self.output_dir)

    # =====================================================================
    #  内部工具
    # =====================================================================

    def _count_attempt(self) -> int:
        self._total_attempts += 1
        if self._total_attempts > _MAX_TOTAL_ATTEMPTS:
            raise MDError(f"已达全局最大尝试次数 ({_MAX_TOTAL_ATTEMPTS}), 终止运行")
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
            raise MDError(
                f"命令失败: {' '.join(full_cmd)}",
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e
        except FileNotFoundError:
            raise MDError(
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
            raise MDError(
                f"拓扑文件不存在: {self.topol_path}\n"
                f"请先运行 01_pdb2gmx.py 完成准备步骤"
            )

        # 验证输入目录
        if not self.input_dir.is_dir():
            raise MDError(
                f"输入目录不存在: {self.input_dir}\n"
                f"请先运行 04_npt.py 完成 NPT 压力平衡"
            )

        # 自动查找 npt.gro
        npt_gro_path = self.input_dir / "npt.gro"
        if npt_gro_path.is_file():
            self.npt_gro = npt_gro_path
        else:
            gro_files = sorted(self.input_dir.glob("*.gro"))
            if gro_files:
                self.npt_gro = gro_files[0]
                warn(f"未找到 npt.gro, 将使用: {self.npt_gro.name}")
            else:
                raise MDError(
                    f"未找到 .gro 文件于 {self.input_dir}\n"
                    f"请确认 NPT 压力平衡步骤已完成"
                )

        # 自动查找 npt.cpt (checkpoint 文件, 用于 continuation)
        npt_cpt_path = self.input_dir / "npt.cpt"
        if npt_cpt_path.is_file():
            self.npt_cpt = npt_cpt_path
        else:
            cpt_files = sorted(self.input_dir.glob("*.cpt"))
            if cpt_files:
                self.npt_cpt = cpt_files[0]
                warn(f"未找到 npt.cpt, 将使用: {self.npt_cpt.name}")
            else:
                warn("未找到 .cpt checkpoint 文件, grompp 将不使用 -t 参数")
                warn("成品 MD 需要从 NPT 继续, 缺少 checkpoint 可能导致速度不连续")

        info(f"输入目录:       {self.input_dir}")
        info(f"拓扑文件:       {self.topol_path}")
        info(f"NPT 结构文件:   {self.npt_gro}")
        if self.npt_cpt:
            info(f"NPT 检查点:     {self.npt_cpt}")

    # =====================================================================
    #  grompp + mdrun
    # =====================================================================

    def _run_grompp(self, mdp_path: Path, attempt_label: str) -> Path:
        """执行 grompp 生成 md.tpr, -maxwarn 递增重试。

        成品 MD grompp: 使用 -t 指定 NPT checkpoint, 不使用 -r 位置约束。

        Returns:
            生成的 md.tpr 路径
        """
        tpr_path = self.output_dir / "md.tpr"

        # 构建基础命令参数 (成品 MD 不用 -r)
        common_args = [
            "-f", str(mdp_path),
            "-c", str(self.npt_gro),
            "-p", str(self.topol_path),
            "-o", str(tpr_path),
        ]

        # 如果有 checkpoint 文件, 添加 -t 参数
        if self.npt_cpt is not None:
            common_args.extend(["-t", str(self.npt_cpt)])

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
            except MDError as e:
                warn(f"  grompp 失败 (maxwarn={maxwarn})")
                if maxwarn == _MAXWARN_SEQUENCE[-1]:
                    raise MDError(
                        f"grompp ({attempt_label}): 所有 -maxwarn 均已尝试完毕",
                        stderr=e.stderr, stdout=e.stdout,
                    ) from e

        raise MDError(f"grompp ({attempt_label}): 未能生成 md.tpr")

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
                        "mdrun", "-deffnm", "md", "-v", "-nb", "gpu",
                        cwd=self.output_dir,
                    )
                except MDError as e:
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
                    "mdrun", "-deffnm", "md", "-v",
                    cwd=self.output_dir,
                )
            except MDError as e:
                warn(f"  CPU mdrun 失败 (第 {retry + 1} 次)")
                if retry == _MDRUN_CPU_RETRY - 1:
                    raise

        raise MDError("mdrun: 所有重试均已用尽")

    # =====================================================================
    #  单次 MD 尝试
    # =====================================================================

    def _try_md_once(
        self, dt: float, nsteps: int, tau_p: float, tau_t: float,
        attempt_label: str,
    ) -> Tuple[bool, MDQualityResult, Path]:
        """执行一次完整的成品 MD (grompp + mdrun) 并检测轨迹质量。

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
        adjusted_mdp = self.adjusted_mdp_dir / f"md_{attempt_label}.mdp"
        _create_adjusted_mdp(template, dt, nsteps, tau_p, tau_t, adjusted_mdp)
        sim_ns = dt * nsteps / 1000.0
        info(f"  使用 MDP: {adjusted_mdp.name}  "
             f"(dt={dt}, nsteps={nsteps} [{sim_ns:.0f} ns], "
             f"tau_p={tau_p:.1f}, tau_t={tau_t:.2f})")

        # ---- grompp ----
        self._run_grompp(adjusted_mdp, attempt_label)

        # ---- mdrun ----
        info(f"  运行 mdrun (成品 MD {'GPU' if self.gpu else 'CPU'}, "
             f"预计 {sim_ns:.0f} ns) ...")
        try:
            self._run_mdrun()
        except MDError as e:
            # mdrun 崩溃 (如 LINCS error 导致直接退出)
            quality = MDQualityResult()
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

        # ---- 分析 MD 质量 ----
        log_path = self.output_dir / "md.log"
        # 目标温度/压力必须来自实际使用的 mdp (绝不硬编码兜底)
        temp_target, pres_target = _read_mdp_targets(
            adjusted_mdp.read_text(encoding="utf-8")
        )
        if temp_target is None:
            raise MDConfigError(
                f"无法从 {adjusted_mdp.name} 读取目标温度 (ref-t / gen-temp 均缺失),\n"
                f"请检查模板 {_MDP_TEMPLATE_DIR}/{_MD_MDP_TEMPLATE} 是否包含 ref-t"
            )
        if pres_target is None:
            raise MDConfigError(
                f"无法从 {adjusted_mdp.name} 读取目标压力 (ref-p 缺失),\n"
                f"请检查模板 {_MDP_TEMPLATE_DIR}/{_MD_MDP_TEMPLATE} 是否包含 ref-p"
            )
        quality = _parse_md_log(
            log_path, expected_nsteps=nsteps,
            temp_target=temp_target, pres_target=pres_target,
        )

        # 用我们设置的参数覆盖
        quality.dt = dt
        quality.total_steps = nsteps
        if quality.sim_time is None:
            quality.sim_time = dt * nsteps

        # ---- 检查轨迹文件 ----
        xtc_path = self.output_dir / "md.xtc"
        if xtc_path.is_file():
            quality.traj_file_exists = True
            quality.traj_file_size_mb = xtc_path.stat().st_size / (1024 * 1024)

        # ---- 额外检查: 从 .edr 文件提取时序数据 ----
        edr_path = self.output_dir / "md.edr"
        if edr_path.is_file():
            # 提取温度时序
            if not quality.temp_values:
                _, temp_vals = _extract_energy_series(
                    edr_path, self.gmx_cmd, "Temperature", self.dry_run,
                )
                if temp_vals:
                    quality.temp_values = temp_vals
                    # .edr 是本步骤的权威数据源: 逐帧真实温度直接来自能量文件,
                    # 不经过 md.log 的排版解析。因此它的均值优先于统计块。
                    # (统计块解析依赖 GROMACS 的输出排版, 跨版本会变 —— 见
                    #  _stats_block_mean。)
                    quality.temp_average = sum(temp_vals) / len(temp_vals)
                    # edr 为权威来源: 均值被刷新后需重新与目标温度比对
                    if abs(quality.temp_average - temp_target) > 20.0:
                        quality.errors.append(
                            f"温度偏差过大: 平均={quality.temp_average:.1f}K, "
                            f"目标={temp_target:.1f}K, "
                            f"偏差={abs(quality.temp_average - temp_target):.1f}K"
                        )
                        quality.stable = False
                    n_temp = len(temp_vals)
                    if n_temp >= 20:
                        front_n = max(1, n_temp // 5)
                        front_avg = sum(temp_vals[:front_n]) / front_n
                        back_avg = sum(temp_vals[-front_n:]) / front_n
                        quality.temp_drift = back_avg - front_avg
                        if abs(quality.temp_drift) > 40.0:
                            quality.errors.append(
                                f"温度漂移过大: {quality.temp_drift:+.1f}K"
                            )
                            quality.stable = False

            # 提取压力时序
            if not quality.pres_values:
                _, pres_vals = _extract_energy_series(
                    edr_path, self.gmx_cmd, "Pressure", self.dry_run,
                )
                if pres_vals:
                    quality.pres_values = pres_vals
                    quality.pres_average = sum(pres_vals) / len(pres_vals)
                    # edr 为权威来源: 均值被刷新后需重新与目标压力比对
                    if abs(quality.pres_average - pres_target) > 100.0:
                        quality.errors.append(
                            f"压力偏差过大: 平均={quality.pres_average:.1f}bar, "
                            f"目标={pres_target:.1f}bar, "
                            f"偏差={abs(quality.pres_average - pres_target):.1f}bar"
                        )
                        quality.stable = False
                    n_pres = len(pres_vals)
                    if n_pres >= 20:
                        front_n = max(1, n_pres // 5)
                        front_avg = sum(pres_vals[:front_n]) / front_n
                        back_avg = sum(pres_vals[-front_n:]) / front_n
                        quality.pres_drift = back_avg - front_avg
                        if abs(quality.pres_drift) > 200.0:
                            quality.errors.append(
                                f"压力漂移过大: {quality.pres_drift:+.1f}bar"
                            )
                            quality.stable = False

            # 提取势能时序
            if not quality.pot_values:
                _, pot_vals = _extract_energy_series(
                    edr_path, self.gmx_cmd, "Potential", self.dry_run,
                )
                if pot_vals:
                    quality.pot_values = pot_vals
                    n_pot = len(pot_vals)
                    if n_pot >= 20:
                        front_n = max(1, n_pot // 5)
                        quality.pot_initial = sum(pot_vals[:front_n]) / front_n
                        quality.pot_final = sum(pot_vals[-front_n:]) / front_n
                        if abs(quality.pot_initial) > 1e-10:
                            drift = (
                                (quality.pot_final - quality.pot_initial)
                                / abs(quality.pot_initial) * 100
                            )
                            quality.pot_drift_percent = drift
                            if abs(drift) > 20.0:
                                quality.errors.append(
                                    f"势能漂移过大: {drift:+.1f}% (前后 20% 段差异 > 20%)"
                                )
                                quality.stable = False

            # 提取密度时序
            if not quality.dens_values:
                _, dens_vals = _extract_energy_series(
                    edr_path, self.gmx_cmd, "Density", self.dry_run,
                )
                if dens_vals:
                    quality.dens_values = dens_vals
                    n_dens = len(dens_vals)
                    if n_dens >= 20:
                        quality.dens_average = sum(dens_vals) / n_dens
                        front_n = max(1, n_dens // 5)
                        front_avg = sum(dens_vals[:front_n]) / front_n
                        back_avg = sum(dens_vals[-front_n:]) / front_n
                        if abs(front_avg) > 1e-10:
                            quality.dens_drift_percent = (
                                (back_avg - front_avg) / abs(front_avg) * 100
                            )
                            if abs(quality.dens_drift_percent) > 10.0:
                                quality.errors.append(
                                    f"密度漂移过大: {quality.dens_drift_percent:+.1f}%"
                                )
                                quality.stable = False

        # 综合判断
        success = (
            quality.stable
            and not quality.has_lincs_warning
            and not quality.has_nan_inf
            and not quality.has_step_size_error
            and len(quality.errors) == 0
        )

        return success, quality, adjusted_mdp

    # =====================================================================
    #  主流程
    # =====================================================================

    def run(self) -> dict:
        """执行成品 MD, 不稳定时自动调整 dt/nsteps/tau_p/tau_t 重试。

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
                sim_time         — float  模拟时长 (ps)
                traj_file        — str    轨迹文件路径
                output_dir       — str    输出目录
                report_path      — str    遍历报告路径
                total_attempts   — int    总尝试次数
                error            — str    (仅失败时)
        """
        banner("GROMACS 成品模拟 (Production MD) — 自动控制脚本")

        try:
            # ---- 前置检查 ----
            self._detect_inputs()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adjusted_mdp_dir.mkdir(parents=True, exist_ok=True)
            info(f"输出目录: {self.output_dir}")

            # ---- 遍历 MD 参数 ----
            tried_params: set = set()
            final_quality: Optional[MDQualityResult] = None
            final_dt: float = 0.0
            final_nsteps: int = 0
            final_tau_p: float = 0.0
            final_tau_t: float = 0.0

            for dt, nsteps, tau_p, tau_t, desc in _MD_PARAM_TRAVERSE:
                param_key = (dt, nsteps, tau_p, tau_t)
                if param_key in tried_params:
                    continue
                tried_params.add(param_key)

                attempt_idx = len(self.report.entries) + 1
                attempt_label = f"attempt_{attempt_idx}"

                sim_ns = dt * nsteps / 1000.0
                info(f"\n{'─' * 50}")
                info(f"尝试 [{attempt_idx}]: {desc}  "
                     f"(dt={dt}, nsteps={nsteps} [{sim_ns:.0f} ns], "
                     f"tau_p={tau_p:.1f}, tau_t={tau_t:.2f})")

                t0 = time.time()
                try:
                    success, quality, mdp_path = self._try_md_once(
                        dt, nsteps, tau_p, tau_t, attempt_label,
                    )
                except MDConfigError:
                    # 配置/解析类错误 — 换参数无法解决, 立即终止阶梯
                    raise
                except MDError as e:
                    dt_elapsed = time.time() - t0
                    warn(f"MD 尝试失败 (命令执行错误): {e}")
                    self.report.add_attempt(
                        step="md",
                        status="failed",
                        dt=dt,
                        nsteps=nsteps,
                        tau_p=tau_p,
                        tau_t=tau_t,
                        mdp_file=str(
                            self.adjusted_mdp_dir / f"md_{attempt_label}.mdp"
                        ),
                        errors=[str(e)],
                        error_summary=self._extract_error_summary(
                            e.stderr, e.stdout
                        ),
                        duration=dt_elapsed,
                    )
                    continue

                dt_elapsed = time.time() - t0

                sim_time_ns = (dt * nsteps / 1000.0) if dt and nsteps else "?"

                # 构建质量详情
                quality_detail = {
                    "completed": "是" if quality.completed else "否",
                    "completed_steps": (
                        f"{quality.completed_steps}/{nsteps}"
                        if quality.completed_steps > 0 else "?"
                    ),
                    "sim_time": (
                        f"{quality.sim_time:.0f}"
                        if quality.sim_time else "?"
                    ),
                    "sim_time_ns": (
                        f"{quality.sim_time / 1000:.1f}"
                        if quality.sim_time else f"{sim_time_ns}"
                    ),
                    "traj_file": (
                        f"md.xtc ({quality.traj_file_size_mb:.1f} MB)"
                        if quality.traj_file_exists else "未生成"
                    ),
                    "traj_size_mb": (
                        f"{quality.traj_file_size_mb:.1f}"
                        if quality.traj_file_size_mb else "?"
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
                    info(f"[成功] 成品 MD 完成 — {desc}")
                    info(f"  温度:  {quality_detail['temp_average']} K  "
                         f"(目标: {quality_detail['temp_target']} K, "
                         f"RMSD: {quality_detail['temp_rmsd']} K)")
                    info(f"  压力:  {quality_detail['pres_average']} bar  "
                         f"(目标: {quality_detail['pres_target']} bar)")
                    info(f"  密度:  {quality_detail['dens_average']} kg/m³")
                    info(f"  轨迹:  {quality_detail['traj_file']}")
                    info(f"  耗时:  {dt_elapsed / 60:.1f} min ({dt_elapsed:.1f}s)")

                    self.report.add_attempt(
                        step="md",
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
                    warn(f"[失败] 成品 MD 不稳定 — {desc}")
                    if quality.errors:
                        for e_msg in quality.errors:
                            warn(f"  ! {e_msg}")
                    self.report.add_attempt(
                        step="md",
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
            if final_quality is not None and final_quality.stable:
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
                    "模拟时长 (ns)": (
                        f"{final_quality.sim_time / 1000:.1f}"
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
                    "轨迹文件": (
                        f"md.xtc ({final_quality.traj_file_size_mb:.1f} MB)"
                        if final_quality.traj_file_exists else "未生成"
                    ),
                    "总耗时": (
                        f"{(datetime.now() - self.report.start_time).total_seconds():.1f}s"
                    ),
                }
                self._collect_files()
                report_path = self.report.save()

                banner("MD 完成 — 成品模拟成功")
                info(f"遍历报告: {report_path}")
                info(f"轨迹文件: {self.output_dir / 'md.xtc'}")
                info(f"能量文件: {self.output_dir / 'md.edr'}")
                info("")
                info("后续分析建议:")
                info(f"  gmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -center")
                info(f"  gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd.xvg -tu ns")
                info(f"  gmx gyrate -s md.tpr -f md_noPBC.xtc -o gyrate.xvg")
                info(f"  gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res")
                info(f"  gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond.xvg")

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
                    "sim_time": final_quality.sim_time,
                    "traj_file": str(self.output_dir / "md.xtc"),
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }
            else:
                self.report.success = False
                self._collect_files()
                report_path = self.report.save()

                banner("MD 失败 — 成品模拟未达标")
                error("所有参数组合均已尝试完毕, 轨迹仍不稳定")
                info(f"遍历报告: {report_path}")

                return {
                    "success": False,
                    "error": "MD 失败: 所有 dt/nsteps/tau_p/tau_t 组合均已尝试",
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }

        except MDError as e:
            self.report.success = False
            self.report.final_params["错误信息"] = str(e)
            self._collect_files()
            report_path = self.report.save()

            banner("MD 失败")
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
            "md.tpr", "md.gro", "md.log", "md.edr", "md.trr",
            "md.cpt", "md.xtc", "mdout.mdp",
        ]
        for fname in key_patterns:
            fpath = self.output_dir / fname
            if fpath.is_file():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                collected.append(f"{fname:<30s} {size_mb:>10.1f} MB")

        # adjusted_mdp 下的文件
        if self.adjusted_mdp_dir.is_dir():
            for mdp_file in sorted(self.adjusted_mdp_dir.glob("*.mdp")):
                rel = str(mdp_file.relative_to(self.output_dir))
                collected.append(f"{rel:<30s} {'(参数文件)':>10s}")

        self.report.generated_files = collected


# =========================================================================
#  公共 API — 供 run_md.py 通过 import 调用
# =========================================================================

def run_md(
    input_dir: str = "MD_result/npt",
    output_dir: str = "MD_result/md",
    gmx_cmd: str = "gmx",
    gpu: bool = True,
    dry_run: bool = False,
    topol_dir: Optional[str] = None,
) -> dict:
    """执行 GROMACS 成品模拟 (Production MD)，自动检测轨迹稳定性并调整参数。

    从 NPT 步骤的输出目录读取 npt.gro + npt.cpt，从 topol_dir 读取 topol.top，
    运行 grompp + mdrun，轨迹不稳定时自动调整 dt/nsteps/tau_p/tau_t 重试。

    Args:
        input_dir:  输入目录 (包含 npt.gro + npt.cpt), 默认 MD_result/npt
        output_dir: 输出目录, 默认 MD_result/md
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
            sim_time         — float  模拟时长 (ps)
            traj_file        — str    轨迹文件路径
            output_dir       — str    输出目录
            report_path      — str    traverse_report.txt 路径
            total_attempts   — int    总尝试次数
            error            — str    (仅失败时)

    Example:
        >>> import importlib
        >>> md = importlib.import_module("05_md")
        >>> result = md.run_md()
        >>> print(result["traj_file"])

    在 run_md.py 中集成示例:
        import importlib
        md = importlib.import_module("05_md")
        result = md.run_md(input_dir=str(self.result_dir / "npt"),
                           topol_dir="MD_result/pdb2gmx")
        if not result["success"]:
            raise MDError(1, stderr=result.get("error", ""))
    """
    runner = MDRunner(
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
        prog="05_md.py",
        description="GROMACS 成品模拟 (Production MD) — 自动检测轨迹稳定性并调整参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 05_md.py
  python 05_md.py --input MD_result/npt --output MD_result/md
  python 05_md.py --dry-run
  python 05_md.py --no-gpu
        """,
    )
    parser.add_argument(
        "--input", default=DEFAULTS["input_dir"],
        help=f"输入目录 (包含 npt.gro + npt.cpt, 默认: {DEFAULTS['input_dir']})",
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

    result = run_md(
        input_dir=args.input,
        output_dir=args.output,
        gmx_cmd=args.gmx,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )

    if result["success"]:
        print(f"\n{'=' * 60}")
        print(f"  MD 完成 — 成品模拟成功")
        print(f"  温度:  {result['temp_average']} K (目标: {result['temp_target']} K)")
        print(f"  压力:  {result['pres_average']} bar (目标: {result['pres_target']} bar)")
        print(f"  密度:  {result['dens_average']} kg/m³")
        print(f"  轨迹:  {result['traj_file']}")
        print(f"  报告:  {result['report_path']}")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"  MD 失败")
        print(f"  原因:  {result.get('error', '未知')}")
        print(f"  报告:  {result.get('report_path', 'N/A')}")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
