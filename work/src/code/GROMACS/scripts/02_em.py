#!/usr/bin/env python3
"""
GROMACS 能量最小化 (EM) 自动化脚本
自动控制 EM 步骤, 检测收敛质量, 不收敛时自动调整参数重试

检测策略:
  - 解析 mdrun 输出, 检查 Fmax 是否低于 emtol
  - 分析势能 (Epot) 变化曲线, 确认持续下降且无剧烈震荡
  - 检查收敛标志: "Converged to machine precision" / "Energy minimization converged"
  - 检测致命错误: LINCS WARNING / nan / inf / step size too large

不收敛时自动调整参数 (在原 mdp 文件外创建新文件, 原始 em.mdp 不被修改):
  emtol:  1000.0 → 500.0 → 100.0 → 50.0
  nsteps: 50000 → 100000 → 200000 → 500000

输入:  MD_result/pdb2gmx/  (从准备步骤自动读取 topol.top + *_solv_ions.gro)
输出:  MD_result/em/
  ├── em.tpr / em.gro / em.log / em.edr / em.trr
  ├── adjusted_mdp/           # 调整后的参数文件 (原始 em.mdp 不被修改)
  │   ├── em_attempt_1.mdp
  │   └── ...
  └── traverse_report.txt     # 遍历报告

用法:
  # 独立脚本
  python 02_em.py

  # 被 run_md.py 导入
  import importlib
  em = importlib.import_module("02_em")
  result = em.run_em()
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
    "input_dir": "MD_result/pdb2gmx",
    "output_dir": "MD_result/em",
    "gmx": "gmx",
    "dry_run": False,
}

# MDP 模板路径 (项目根目录下)
_MDP_TEMPLATE_DIR = "mdp_files"
_EM_MDP_TEMPLATE = "em.mdp"

# grompp -maxwarn 递增序列
_MAXWARN_SEQUENCE: List[int] = [10, 20, 50, 100]

# GPU → CPU 降级重试
_MDRUN_GPU_RETRY = 3
_MDRUN_CPU_RETRY = 3

# 全局最大尝试次数
_MAX_TOTAL_ATTEMPTS = 100


# ==================== EM 参数遍历策略 ====================

# 不收敛时的 emtol + nsteps 组合遍历顺序
# 原则: 先增加步数, 再逐步收紧容差配合更大步数
_EM_PARAM_TRAVERSE: List[Tuple[float, int, str]] = [
    # (emtol, nsteps, 说明)
    (2000.0, 50000,   "默认参数"),
    (2000.0, 100000,  "增加步数 ×2"),
    (2000.0, 200000,  "增加步数 ×4"),
    (1000.0, 200000,  "收紧容差 + 步数 ×4"),
    (1000.0, 500000,  "收紧容差 + 最大步数"),
    (200.0,  500000,  "严格容差 + 最大步数"),
    (100.0,  500000,  "最严格容差 + 最大步数"),
]


# ==================== 异常类 ====================

class EMError(Exception):
    """EM 步骤中的错误。"""

    def __init__(self, message: str, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


# ==================== 收敛检测 ====================

class ConvergenceResult:
    """EM 收敛检测结果。"""

    def __init__(self) -> None:
        self.converged: bool = False
        self.converged_to_machine_precision: bool = False
        self.fmax_final: Optional[float] = None
        self.fmax_target: Optional[float] = None
        self.epot_initial: Optional[float] = None
        self.epot_final: Optional[float] = None
        self.total_steps: int = 0
        self.epot_values: List[float] = []          # 每一步的 Epot
        self.epot_trend_ok: bool = False             # 能量持续下降
        self.epot_oscillation_count: int = 0         # 剧烈震荡步数
        self.epot_increase_count: int = 0            # 能量上升步数
        self.has_lincs_warning: bool = False
        self.has_nan_inf: bool = False
        self.has_step_size_error: bool = False
        self.convergence_message: str = ""
        self.errors: List[str] = []                  # 所有检测到的错误


def _parse_em_log(log_path: Path) -> ConvergenceResult:
    """解析 GROMACS EM 的 .log 文件, 检测收敛质量。

    检查项目:
      1. 收敛标志: "Converged to machine precision" / "Energy minimization converged"
      2. Fmax 是否低于 emtol
      3. 势能是否持续下降, 有无剧烈震荡
      4. 致命错误: LINCS WARNING / nan / inf / step size too large

    Args:
        log_path: em.log 文件路径

    Returns:
        ConvergenceResult 包含所有检测结果
    """
    result = ConvergenceResult()

    if not log_path.is_file():
        result.errors.append("日志文件不存在")
        return result

    content = log_path.read_text(encoding="utf-8", errors="replace")

    # ---- 1. 致命错误检测 ----

    if re.search(r'LINCS\s+WARNING', content, re.IGNORECASE):
        result.has_lincs_warning = True
        result.errors.append("LINCS WARNING — 约束算法失败, 体系中存在严重冲突")

        # nan/inf 检测: 只在模拟步骤输出行中搜索
    # (em.log 参数段含 "epsilon-rf = inf" 等合法值, 不能盲目全文搜索)
    _step_lines = [l for l in content.splitlines()
                   if re.search(r'Step=', l) or re.search(r'Epot=', l)]
    _step_text = "\n".join(_step_lines)
    if re.search(r'\bnan\b', _step_text, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("nan 值 — 模拟崩溃")
    if re.search(r'\binf\b', _step_text, re.IGNORECASE):
        result.has_nan_inf = True
        result.errors.append("inf 值 — 模拟崩溃")

    if re.search(r'step\s+size\s+too\s+large', content, re.IGNORECASE):
        result.has_step_size_error = True
        result.errors.append("Step size too large — 力场梯度过大, 体系不稳定")

    # ---- 2. 收敛标志 ----
    # GROMACS SD (Steepest Descents) 收敛格式:
    #   "Steepest Descents converged to Fmax < 100 in 4782 steps"
    # 也匹配传统格式:
    #   "Converged to machine precision" / "Energy minimization converged"

    m = re.search(
        r'(?:steepest\s+descents\s+converged\s+to\s+Fmax'
        r'|converged\s+to\s+machine\s+precision'
        r'|energy\s+minimization\s+converged)',
        content, re.IGNORECASE,
    )
    if m:
        result.converged_to_machine_precision = True
        result.convergence_message = m.group(0)
        result.converged = True

    # "未收敛" 覆盖 — 仅在 GROMACS 尚未报告收敛时生效
    if not result.converged:
        if re.search(r'did\s+(not|NOT)\s+converge|has\s+not\s+converged|not\s+converged',
                     content, re.IGNORECASE):
            result.converged = False
            result.converged_to_machine_precision = False

    # ---- 3. 解析每步的 Epot 和 Fmax ----
    # GROMACS 输出格式:
    #   Step=    0, Dmax= 1.0e-02 nm, Epot= -1.23456e+05 Fmax= 5.67890e+08, atom= 1234
    step_pattern = re.compile(
        r'Step=\s*(\d+).*?Epot=\s*([+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)'
        r'.*?Fmax=\s*([+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE,
    )

    epot_list: List[Tuple[int, float, float]] = []  # (step, epot, fmax)
    for m in step_pattern.finditer(content):
        step = int(m.group(1))
        try:
            epot_val = float(m.group(2))
            fmax_val = float(m.group(3))
            epot_list.append((step, epot_val, fmax_val))
        except ValueError:
            continue

    if epot_list:
        result.epot_values = [e for _, e, _ in epot_list]
        result.epot_initial = epot_list[0][1]
        result.epot_final = epot_list[-1][1]
        result.fmax_final = epot_list[-1][2]
        result.total_steps = max(result.total_steps, len(epot_list))

        # ---- 分析 Epot 下降趋势 ----
        epots = result.epot_values
        n = len(epots)

        epot_range = abs(epots[0] - epots[-1])

        # 条件 1: 最终能量 < 初始能量 (或初始能量 ≈ 0 的特殊情况)
        if epots[-1] >= epots[0] and epot_range > 1e-6:
            result.errors.append(
                f"势能未下降 (初始={epots[0]:.4e}, 最终={epots[-1]:.4e})"
            )

        # 条件 2: 检查剧烈震荡 —— 统计能量上升步数和幅度
        increase_steps = 0
        oscillation_steps = 0
        for i in range(1, n):
            delta = epots[i] - epots[i - 1]
            if delta > 0:
                increase_steps += 1
                # 如果单步上升幅度超过整体下降幅度的 5%, 视为异常震荡
                if epot_range > 0 and delta > 0.05 * epot_range:
                    oscillation_steps += 1

        result.epot_increase_count = increase_steps
        result.epot_oscillation_count = oscillation_steps

        # 判断: 上升步数 ≤ 总步数的 10%, 且异常震荡数 ≤ 5%
        if increase_steps > 0:
            increase_ratio = increase_steps / n
            if increase_ratio > 0.10:
                result.errors.append(
                    f"势能上升步数过多 ({increase_steps}/{n} = {increase_ratio:.1%})"
                )
            if oscillation_steps > max(1, n * 0.05):
                result.errors.append(
                    f"势能剧烈震荡 ({oscillation_steps} 步单步上升 > 总体下降的 5%)"
                )

        if increase_steps <= max(1, n * 0.10) and oscillation_steps <= max(1, n * 0.05):
            result.epot_trend_ok = True

    # ---- 4. 检查 Fmax vs emtol ----
    tol_match = re.search(
        r'Tolerance\s*\(Fmax\)\s*=\s*([+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)',
        content, re.IGNORECASE,
    )
    if tol_match:
        result.fmax_target = float(tol_match.group(1))

    if result.fmax_final is not None and result.fmax_target is not None:
        if result.fmax_final > result.fmax_target:
            result.converged = False
            result.errors.append(
                f"Fmax ({result.fmax_final:.4e}) > 容差 ({result.fmax_target:.4e}), 未收敛"
            )

    # ---- 5. 综合判断 ----
    # 即使 GROMACS 报告 "converged", 如果有无声错误也判定为失败
    if result.has_lincs_warning or result.has_nan_inf or result.has_step_size_error:
        result.converged = False

    return result


# ==================== MDP 参数文件管理 ====================

def _read_template_mdp(project_root: Path) -> str:
    """读取原始 em.mdp 模板内容。"""
    template_path = project_root / _MDP_TEMPLATE_DIR / _EM_MDP_TEMPLATE
    if not template_path.is_file():
        raise EMError(
            f"EM 参数模板文件不存在: {template_path}\n"
            f"请确认 {_MDP_TEMPLATE_DIR}/{_EM_MDP_TEMPLATE} 位于项目根目录"
        )
    return template_path.read_text(encoding="utf-8")


def _create_adjusted_mdp(
    template_content: str,
    emtol: float,
    nsteps: int,
    output_path: Path,
) -> Path:
    """基于模板内容创建调整后的 mdp 文件 (不修改原始文件)。

    Args:
        template_content: 原始 em.mdp 的文本内容
        emtol: 新的能量容差
        nsteps: 新的最大步数
        output_path: 输出文件路径

    Returns:
        创建的文件路径
    """
    content = template_content

    # 替换 emtol 行
    content = re.sub(
        r'^(\s*emtol\s*=\s*)(.*)$',
        f'emtol       = {emtol:.1f}',
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

    # 添加注释标记 (如果还没有)
    if "auto-adjusted" not in content.lower():
        content = (
            f"; 自动调整的 EM 参数 (由 02_em.py 生成)\n"
            f"; emtol={emtol:.1f}  nsteps={nsteps}\n"
            + content
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ==================== 遍历报告 ====================

class TraverseReport:
    """EM 遍历报告, 记录每次尝试的详细信息和收敛分析。"""

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
        lines.append("GROMACS 能量最小化 (EM) — 自动遍历报告")
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
        lines.append("参数遍历策略 (不收敛时自动调整)")
        lines.append(sep)
        lines.append(f"  默认 MDP 模板: {_MDP_TEMPLATE_DIR}/{_EM_MDP_TEMPLATE}")
        lines.append(f"  调整参数:      emtol, nsteps")
        lines.append(f"  原始模板不会被修改, 每次尝试生成新的 mdp 文件")
        lines.append("")
        for i, (emtol, nsteps, desc) in enumerate(_EM_PARAM_TRAVERSE, 1):
            lines.append(f"  [{i}] emtol={emtol:.1f}, nsteps={nsteps:>6d}  — {desc}")
        lines.append("")

        # ---- 遍历详情 ----
        lines.append(sep)
        lines.append("遍历详情")
        lines.append(sep)

        for i, entry in enumerate(self.entries, 1):
            status = "成功" if entry.get("status") == "success" else "失败"
            icon = "OK" if status == "成功" else "!!"
            lines.append(f"\n  [{icon}] 尝试 {i}  |  {entry['timestamp']}")
            lines.append(f"       参数:     emtol={entry.get('emtol', '?')}, "
                         f"nsteps={entry.get('nsteps', '?')}")
            lines.append(f"       结果:     {status}")

            conv = entry.get("convergence", {})
            if conv:
                lines.append(f"       ---- 收敛检测 ----")
                lines.append(f"       收敛标志: {conv.get('convergence_message', '无')}")
                lines.append(f"       Fmax 目标:  {conv.get('fmax_target', '?')}")
                lines.append(f"       Fmax 实际:  {conv.get('fmax_final', '?')}")
                lines.append(f"       Epot 初始:  {conv.get('epot_initial', '?')}")
                lines.append(f"       Epot 最终:  {conv.get('epot_final', '?')}")
                lines.append(f"       Epot 下降:  {conv.get('epot_delta', '?')}")
                lines.append(f"       总步数:     {conv.get('total_steps', '?')}")
                lines.append(f"       能量上升步: {conv.get('epot_increase_count', '?')}"
                             f" (剧烈震荡: {conv.get('epot_oscillation_count', '?')})")
                lines.append(f"       能量趋势:   {conv.get('epot_trend', '?')}")
                if conv.get("lincs_warning"):
                    lines.append(f"       [致命] LINCS WARNING")
                if conv.get("nan_inf"):
                    lines.append(f"       [致命] nan/inf 值")
                if conv.get("step_size_error"):
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


# ==================== EM 运行器 ====================

class EMRunner:
    """GROMACS 能量最小化 (EM) 运行器。

    读取准备步骤的输出文件, 执行 grompp + mdrun,
    检测收敛质量, 不收敛时自动调整 emtol/nsteps 重试。

    用法:
        runner = EMRunner()
        result = runner.run()
    """

    def __init__(
        self,
        input_dir: str = "MD_result/pdb2gmx",
        output_dir: str = "MD_result/em",
        gmx_cmd: str = "gmx",
        gpu: bool = True,
        dry_run: bool = False,
    ) -> None:
        # ---- 路径 ----
        self.project_root = Path(__file__).parent.parent.resolve()
        _input = Path(input_dir)
        self.input_dir = _input.resolve() if _input.is_absolute() else (self.project_root / _input).resolve()
        _output = Path(output_dir)
        self.output_dir = _output.resolve() if _output.is_absolute() else (self.project_root / _output).resolve()
        self.adjusted_mdp_dir = self.output_dir / "adjusted_mdp"

        # ---- 配置 ----
        self.gmx_cmd = gmx_cmd
        self.gpu = gpu
        self.dry_run = dry_run

        # ---- 自动检测 ----
        self.pdb_basename: str = ""
        self.topol_path: Path = self.input_dir / "topol.top"
        self.solv_ions_gro: Optional[Path] = None

        # ---- 运行时状态 ----
        self._total_attempts: int = 0
        self.report = TraverseReport(str(self.input_dir), self.output_dir)

    # =====================================================================
    #  内部工具
    # =====================================================================

    def _count_attempt(self) -> int:
        self._total_attempts += 1
        if self._total_attempts > _MAX_TOTAL_ATTEMPTS:
            raise EMError(f"已达全局最大尝试次数 ({_MAX_TOTAL_ATTEMPTS}), 终止运行")
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
            raise EMError(
                f"命令失败: {' '.join(full_cmd)}",
                stderr=stderr_text,
                stdout=stdout_text,
            ) from e
        except FileNotFoundError:
            raise EMError(
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
        """自动检测输入目录中的文件并提取 basename。"""
        if not self.input_dir.is_dir():
            raise EMError(
                f"输入目录不存在: {self.input_dir}\n"
                f"请先运行 01_pdb2gmx.py 完成准备步骤"
            )

        # 验证 topol.top
        if not self.topol_path.is_file():
            raise EMError(
                f"拓扑文件不存在: {self.topol_path}\n"
                f"请确认 {self.input_dir} 中有 topol.top"
            )

        # 自动查找 *_solv_ions.gro
        gro_files = sorted(self.input_dir.glob("*_solv_ions.gro"))
        if not gro_files:
            raise EMError(
                f"未找到 *_solv_ions.gro 文件于 {self.input_dir}\n"
                f"请确认准备步骤 (genion) 已完成"
            )
        if len(gro_files) > 1:
            warn(f"检测到多个 solv_ions.gro 文件: {[f.name for f in gro_files]}")
            warn(f"将使用第一个: {gro_files[0].name}")

        self.solv_ions_gro = gro_files[0]
        # 从文件名提取 basename (e.g. "protein" from "protein_solv_ions.gro")
        self.pdb_basename = gro_files[0].name.replace("_solv_ions.gro", "")

        info(f"输入目录:       {self.input_dir}")
        info(f"体系名称:       {self.pdb_basename}")
        info(f"拓扑文件:       {self.topol_path}")
        info(f"结构文件:       {self.solv_ions_gro}")

    # =====================================================================
    #  grompp + mdrun
    # =====================================================================

    def _run_grompp(self, mdp_path: Path, attempt_label: str) -> Path:
        """执行 grompp 生成 em.tpr, -maxwarn 递增重试。

        Returns:
            生成的 em.tpr 路径
        """
        tpr_path = self.output_dir / "em.tpr"

        for maxwarn in _MAXWARN_SEQUENCE:
            self._count_attempt()
            if maxwarn > _MAXWARN_SEQUENCE[0]:
                info(f"  grompp 重试: -maxwarn {maxwarn}")

            try:
                self.run_gmx(
                    "grompp",
                    "-f", str(mdp_path),
                    "-c", str(self.solv_ions_gro),
                    "-p", str(self.topol_path),
                    "-o", str(tpr_path),
                    "-maxwarn", str(maxwarn),
                    cwd=self.output_dir,
                )
                info(f"  grompp 完成 → {tpr_path}")
                return tpr_path
            except EMError as e:
                warn(f"  grompp 失败 (maxwarn={maxwarn})")
                if maxwarn == _MAXWARN_SEQUENCE[-1]:
                    raise EMError(
                        f"grompp ({attempt_label}): 所有 -maxwarn 均已尝试完毕",
                        stderr=e.stderr, stdout=e.stdout,
                    ) from e

        raise EMError(f"grompp ({attempt_label}): 未能生成 em.tpr")

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
                        "mdrun", "-deffnm", "em", "-v", "-nb", "gpu",
                        cwd=self.output_dir,
                    )
                except EMError as e:
                    combined = (e.stderr + e.stdout).lower()
                    # 如果是模拟本身的问题 (不是 GPU), 不要重试 GPU, 直接向上抛
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
                    "mdrun", "-deffnm", "em", "-v",
                    cwd=self.output_dir,
                )
            except EMError as e:
                warn(f"  CPU mdrun 失败 (第 {retry + 1} 次)")
                if retry == _MDRUN_CPU_RETRY - 1:
                    raise

        raise EMError("mdrun: 所有重试均已用尽")

    # =====================================================================
    #  单次 EM 尝试
    # =====================================================================

    def _try_em_once(
        self, emtol: float, nsteps: int, attempt_label: str,
    ) -> Tuple[bool, ConvergenceResult, Path]:
        """执行一次完整的 EM (grompp + mdrun) 并检测收敛。

        Args:
            emtol:  能量容差
            nsteps: 最大步数
            attempt_label: 本次尝试的说明标签 (如 "attempt_1")

        Returns:
            (success, convergence_result, mdp_path_used)
        """
        # ---- 创建调整后的 mdp 文件 ----
        template = _read_template_mdp(self.project_root)
        adjusted_mdp = self.adjusted_mdp_dir / f"em_{attempt_label}.mdp"
        _create_adjusted_mdp(template, emtol, nsteps, adjusted_mdp)
        info(f"  使用 MDP: {adjusted_mdp.name}  (emtol={emtol:.1f}, nsteps={nsteps})")

        # ---- grompp ----
        self._run_grompp(adjusted_mdp, attempt_label)

        # ---- mdrun ----
        info(f"  运行 mdrun ...")
        try:
            self._run_mdrun()
        except EMError as e:
            # mdrun 崩溃 (如 LINCS error 导致直接退出)
            conv = ConvergenceResult()
            conv.errors.append(
                f"mdrun 异常退出: {e.stderr[:200] if e.stderr else str(e)}"
            )
            combined = (e.stderr + e.stdout).lower()
            if "lincs" in combined:
                conv.has_lincs_warning = True
            if "nan" in combined or "inf" in combined:
                conv.has_nan_inf = True
            if "step size" in combined:
                conv.has_step_size_error = True
            return False, conv, adjusted_mdp

        # ---- 分析收敛 ----
        log_path = self.output_dir / "em.log"
        conv = _parse_em_log(log_path)

        # 用我们设置的 emtol 覆盖日志中解析到的 (确保一致性)
        if conv.fmax_target is None:
            conv.fmax_target = emtol

        # 额外检查 Fmax 是否低于当前 emtol
        if conv.fmax_final is not None and conv.fmax_final > emtol:
            conv.converged = False
            if not any("Fmax" in e for e in conv.errors):
                conv.errors.append(
                    f"Fmax ({conv.fmax_final:.4e}) > emtol ({emtol:.4e}), 未收敛"
                )

        # ---- 综合判断 ----
        # 致命错误 (体系物理上不可行) — 始终阻塞
        has_fatal = (
            conv.has_lincs_warning
            or conv.has_nan_inf
            or conv.has_step_size_error
        )

        # GROMACS 自身是否报告收敛 (或 Fmax 已达标)
        gromacs_ok = conv.converged_to_machine_precision
        fmax_ok = (
            conv.fmax_final is not None
            and conv.fmax_target is not None
            and conv.fmax_final <= conv.fmax_target
        )

        if has_fatal:
            # 致命错误 → 必须重试
            success = False
        elif gromacs_ok or fmax_ok:
            # GROMACS 已收敛 / Fmax 已达标 → 输出文件有效, 可继续下一步
            # 质量警告 (能量趋势、震荡等) 降级为 ⚠ 提示, 不阻塞流程
            if not gromacs_ok:
                warn("GROMACS 未输出收敛标志, 但 Fmax 已达标 — 视为收敛")
            if conv.errors:
                warn(f"EM 已收敛, 但存在以下质量警告 (不阻塞流程):")
                for e_msg in conv.errors:
                    warn(f"  ⚠ {e_msg}")
            if not conv.epot_trend_ok:
                warn("EM 已收敛, 但势能趋势异常 (不阻塞流程)")
            # 确保 downstream (run() 收尾判决) 能看到收敛状态
            conv.converged = True
            success = True
        else:
            # GROMACS 未收敛且 Fmax 未达标 → 需要尝试下一组参数
            success = False

        return success, conv, adjusted_mdp

    # =====================================================================
    #  主流程
    # =====================================================================

    def run(self) -> dict:
        """执行 EM, 不收敛时自动调整 emtol/nsteps 重试。

        Returns:
            dict:
                success         — bool
                emtol           — float  最终采用的 emtol
                nsteps          — int    最终采用的 nsteps
                fmax_final      — float  最终 Fmax
                epot_initial    — float  初始势能
                epot_final      — float  最终势能
                output_dir      — str    输出目录
                report_path     — str    遍历报告路径
                total_attempts  — int    总尝试次数
                error           — str    (仅失败时)
        """
        banner("GROMACS 能量最小化 (EM) — 自动控制脚本")

        try:
            # ---- 前置检查 ----
            self._detect_inputs()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adjusted_mdp_dir.mkdir(parents=True, exist_ok=True)
            info(f"输出目录: {self.output_dir}")

            # ---- 遍历 EM 参数 ----
            tried_params: set = set()
            final_conv: Optional[ConvergenceResult] = None
            final_emtol: float = 0.0
            final_nsteps: int = 0

            for emtol, nsteps, desc in _EM_PARAM_TRAVERSE:
                param_key = (emtol, nsteps)
                if param_key in tried_params:
                    continue
                tried_params.add(param_key)

                attempt_idx = len(self.report.entries) + 1
                attempt_label = f"attempt_{attempt_idx}"

                info(f"\n{'─' * 50}")
                info(f"尝试 [{attempt_idx}]: {desc}  "
                     f"(emtol={emtol:.1f}, nsteps={nsteps})")

                t0 = time.time()
                try:
                    success, conv, mdp_path = self._try_em_once(
                        emtol, nsteps, attempt_label,
                    )
                except EMError as e:
                    dt = time.time() - t0
                    warn(f"EM 尝试失败 (命令执行错误): {e}")
                    self.report.add_attempt(
                        step="em",
                        status="failed",
                        emtol=emtol,
                        nsteps=nsteps,
                        mdp_file=str(
                            self.adjusted_mdp_dir / f"em_{attempt_label}.mdp"
                        ),
                        errors=[str(e)],
                        error_summary=self._extract_error_summary(
                            e.stderr, e.stdout
                        ),
                        duration=dt,
                    )
                    continue

                dt = time.time() - t0

                # 构建收敛详情
                conv_detail = {
                    "convergence_message": (
                        conv.convergence_message or "(无)"
                    ),
                    "fmax_target": (
                        f"{conv.fmax_target:.4e}"
                        if conv.fmax_target else "?"
                    ),
                    "fmax_final": (
                        f"{conv.fmax_final:.4e}"
                        if conv.fmax_final else "?"
                    ),
                    "epot_initial": (
                        f"{conv.epot_initial:.4e}"
                        if conv.epot_initial else "?"
                    ),
                    "epot_final": (
                        f"{conv.epot_final:.4e}"
                        if conv.epot_final else "?"
                    ),
                    "epot_delta": (
                        f"{conv.epot_initial - conv.epot_final:.4e}"
                        if conv.epot_initial and conv.epot_final else "?"
                    ),
                    "total_steps": conv.total_steps,
                    "epot_increase_count": conv.epot_increase_count,
                    "epot_oscillation_count": conv.epot_oscillation_count,
                    "epot_trend": (
                        "OK (持续下降)" if conv.epot_trend_ok else "异常"
                    ),
                    "lincs_warning": conv.has_lincs_warning,
                    "nan_inf": conv.has_nan_inf,
                    "step_size_error": conv.has_step_size_error,
                }

                if success:
                    info(f"[成功] EM 收敛 — {desc}")
                    if conv.convergence_message:
                        info(f"  GROMACS: {conv.convergence_message}")
                    info(f"  Fmax 最终:   {conv_detail['fmax_final']}  "
                         f"(目标: {conv_detail['fmax_target']})")
                    info(f"  Epot:        {conv_detail['epot_initial']}  "
                         f"→ {conv_detail['epot_final']}")
                    info(f"  耗时:        {dt:.1f}s")

                    self.report.add_attempt(
                        step="em",
                        status="success",
                        emtol=emtol,
                        nsteps=nsteps,
                        convergence=conv_detail,
                        mdp_file=str(mdp_path),
                        duration=dt,
                    )

                    final_conv = conv
                    final_emtol = emtol
                    final_nsteps = nsteps
                    break
                else:
                    warn(f"[失败] EM 未收敛 — {desc}")
                    if conv.errors:
                        for e_msg in conv.errors:
                            warn(f"  ! {e_msg}")
                    self.report.add_attempt(
                        step="em",
                        status="failed",
                        emtol=emtol,
                        nsteps=nsteps,
                        convergence=conv_detail,
                        errors=conv.errors,
                        mdp_file=str(mdp_path),
                        duration=dt,
                    )
                    final_conv = conv
                    final_emtol = emtol
                    final_nsteps = nsteps
                    final_mdp = mdp_path

            # ---- 收尾 ----
            # 分层判定 (与 _try_em_once 一致):
            # GROMACS 收敛 / Fmax 达标 → 视为成功, 质量警告不阻塞
            final_gromacs_ok = (
                final_conv is not None
                and final_conv.converged_to_machine_precision
            )
            final_fmax_ok = (
                final_conv is not None
                and final_conv.fmax_final is not None
                and final_conv.fmax_target is not None
                and final_conv.fmax_final <= final_conv.fmax_target
            )
            final_has_fatal = (
                final_conv is not None
                and (
                    final_conv.has_lincs_warning
                    or final_conv.has_nan_inf
                    or final_conv.has_step_size_error
                )
            )

            if final_conv is not None and (final_conv.converged or final_gromacs_ok or final_fmax_ok) and not final_has_fatal:
                self.report.success = True
                self.report.final_params = {
                    "最终 emtol": f"{final_emtol:.1f}",
                    "最终 nsteps": final_nsteps,
                    "Fmax 最终值": (
                        f"{final_conv.fmax_final:.4e}"
                        if final_conv.fmax_final else "?"
                    ),
                    "Fmax 目标值": (
                        f"{final_conv.fmax_target:.4e}"
                        if final_conv.fmax_target else "?"
                    ),
                    "Epot 初始": (
                        f"{final_conv.epot_initial:.4e}"
                        if final_conv.epot_initial else "?"
                    ),
                    "Epot 最终": (
                        f"{final_conv.epot_final:.4e}"
                        if final_conv.epot_final else "?"
                    ),
                    "总耗时": (
                        f"{(datetime.now() - self.report.start_time).total_seconds():.1f}s"
                    ),
                }
                self._collect_files()
                report_path = self.report.save()

                banner("EM 完成 — 能量最小化收敛")
                info(f"遍历报告: {report_path}")

                return {
                    "success": True,
                    "emtol": final_emtol,
                    "nsteps": final_nsteps,
                    "fmax_final": final_conv.fmax_final,
                    "epot_initial": final_conv.epot_initial,
                    "epot_final": final_conv.epot_final,
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }
            else:
                self.report.success = False
                self._collect_files()
                report_path = self.report.save()

                banner("EM 失败 — 能量最小化未收敛")
                error("所有参数组合均已尝试完毕, 仍未收敛")
                info(f"遍历报告: {report_path}")

                return {
                    "success": False,
                    "error": "EM 未收敛: 所有 emtol/nsteps 组合均已尝试",
                    "output_dir": str(self.output_dir),
                    "report_path": str(report_path),
                    "total_attempts": self._total_attempts,
                }

        except EMError as e:
            self.report.success = False
            self.report.final_params["错误信息"] = str(e)
            self._collect_files()
            report_path = self.report.save()

            banner("EM 失败")
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
            "em.tpr", "em.gro", "em.log", "em.edr", "em.trr",
            "em.cpt", "em.xtc", "mdout.mdp",
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

def run_em(
    input_dir: str = "MD_result/pdb2gmx",
    output_dir: str = "MD_result/em",
    gmx_cmd: str = "gmx",
    gpu: bool = True,
    dry_run: bool = False,
) -> dict:
    """执行 GROMACS 能量最小化 (EM)，自动检测收敛质量并调整参数。

    从准备步骤的输出目录读取 topol.top 和 *_solv_ions.gro，
    运行 grompp + mdrun，不收敛时自动调整 emtol/nsteps 重试。

    Args:
        input_dir:  输入目录 (包含 topol.top + *_solv_ions.gro), 默认 MD_result/pdb2gmx
        output_dir: 输出目录, 默认 MD_result/em
        gmx_cmd:    GROMACS 命令, 默认 "gmx"
        gpu:        是否使用 GPU, 默认 True (失败自动降级 CPU)
        dry_run:    只打印命令不执行

    Returns:
        dict:
            success         — bool
            emtol           — float  最终采用的 emtol
            nsteps          — int    最终采用的 nsteps
            fmax_final      — float  最终 Fmax
            epot_initial    — float  初始势能
            epot_final      — float  最终势能
            output_dir      — str    输出目录
            report_path     — str    traverse_report.txt 路径
            total_attempts  — int    总尝试次数
            error           — str    (仅失败时)

    Example:
        >>> import importlib
        >>> em = importlib.import_module("02_em")
        >>> result = em.run_em()
        >>> print(result["fmax_final"])

    在 run_md.py 中集成示例:
        import importlib
        em = importlib.import_module("02_em")
        result = em.run_em(input_dir=str(self.result_dir / "pdb2gmx"))
        if not result["success"]:
            raise MDError(1, stderr=result.get("error", ""))
    """
    runner = EMRunner(
        input_dir=input_dir,
        output_dir=output_dir,
        gmx_cmd=gmx_cmd,
        gpu=gpu,
        dry_run=dry_run,
    )
    return runner.run()


# =========================================================================
#  CLI 入口 — 作为独立脚本运行
# =========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="02_em.py",
        description="GROMACS 能量最小化 (EM) — 自动检测收敛并调整参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 02_em.py
  python 02_em.py --input MD_result/pdb2gmx --output MD_result/em
  python 02_em.py --dry-run
  python 02_em.py --no-gpu
        """,
    )
    parser.add_argument(
        "--input", default=DEFAULTS["input_dir"],
        help=f"输入目录 (包含 topol.top + *_solv_ions.gro, 默认: {DEFAULTS['input_dir']})",
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

    result = run_em(
        input_dir=args.input,
        output_dir=args.output,
        gmx_cmd=args.gmx,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )

    if result["success"]:
        print(f"\n{'=' * 60}")
        print(f"  EM 完成 — 能量最小化收敛")
        print(f"  Fmax:  {result['fmax_final']}")
        print(f"  Epot:  {result['epot_initial']} → {result['epot_final']}")
        print(f"  报告:  {result['report_path']}")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"  EM 失败")
        print(f"  原因:  {result.get('error', '未知')}")
        print(f"  报告:  {result.get('report_path', 'N/A')}")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
