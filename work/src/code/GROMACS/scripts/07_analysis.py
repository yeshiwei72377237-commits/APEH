#!/usr/bin/env python3
"""
GROMACS MD 后处理分析子脚本 (Step 7) — 结构量 / 表面·溶剂 / 能量分解

在成品模拟 (Step 5 MD) 及归档 (Step 6 Cleanup) 成功完成后由 run_all_md.py 自动调用。

功能 (全部输出为 CSV 表格, 便于 Excel / pandas 直接读取):
  1. 去除周期边界 (trjconv -pbc mol -center)  → md_noPBC.xtc
  2. 结构量:
     - RMSD       (gmx rms)          → rmsd.csv    [time_ns, rmsd_nm]
     - 回转半径   (gmx gyrate)       → gyrate.csv  [time_ps, rg_nm]
     - RMSF       (gmx rmsf -res)    → rmsf.csv    [resid, rmsf_nm]
     - 二级结构   (gmx do_dssp)      → ss_count.csv (可选, 需要 DSSP 外部程序)
  3. 表面·溶剂:
     - SASA       (gmx sasa)         → sasa.csv    [time_ps, area_nm2]
                                      residue_sasa.csv [resid, sasa_nm2]
  4. 能量分解   (gmx energy)         → energy_components.csv
     [time_ps, potential, kinetic, total_energy, temperature, pressure,
      density, coul_sr, lj_sr, coul_14, lj_14, bond, angle, proper_dih]

特性:
  - 分析失败不阻塞主流程 (MD 核心产物已生成, 分析仅为增值输出)
  - 组编号动态解析 (不硬编码 Protein/C-alpha 编号, 与 01_pdb2gmx 一致)
  - 支持 --dry-run 预览模式
  - 输入目录优先取归档目录 output_statistics/<batch>/,
    找不到时回退到 MD_result/md/

调用方式:
  由 run_all_md.py 导入调用:
    analysis = importlib.import_module("07_analysis")
    result = analysis.run_analysis(project_root, stats_dir=..., gmx_cmd="gmx")
  也可独立运行:
    python scripts/07_analysis.py --project-root . --stats-dir output_statistics/<batch>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 兼容独立运行: 将项目根目录加入 sys.path, 使 `from scripts._gmx_utils` 可解析
if __package__ in (None, ""):
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from scripts._gmx_utils import run_gmx


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
    from datetime import datetime
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


# ==================== 常量 ====================

# 分析必需的输入文件 (缺一不可)
_REQUIRED_INPUTS = ["md.tpr", "md.xtc", "md.edr"]

# 能量分解: (CSV 列名, [候选 GROMACS 能量项名称])
# 逐候选尝试, 命中第一个成功者; GROMACS 能量项名在不同版本间略有差异
_ENERGY_TERMS: List[Tuple[str, List[str]]] = [
    ("potential",    ["Potential"]),
    ("kinetic",      ["Kinetic En.", "Kinetic-En.", "Kinetic"]),
    ("total_energy", ["Total Energy", "Total-Energy"]),
    ("temperature",  ["Temperature"]),
    ("pressure",     ["Pressure"]),
    ("density",      ["Density"]),
    ("coul_sr",      ["Coul-SR", "Coulomb (SR)", "Coulomb-SR"]),
    ("lj_sr",        ["LJ-SR", "LJ (SR)"]),
    ("coul_14",      ["Coul-14", "Coulomb-14"]),
    ("lj_14",        ["LJ-14"]),
    ("bond",         ["Bond"]),
    ("angle",        ["Angle"]),
    ("proper_dih",   ["Proper Dih.", "Proper-Dih.", "Proper Dih"]),
]

# 二级结构 (do_dssp -sc) 输出列名 (GROMACS 2020+ 常见顺序, 版本间可能略有差异)
_DSSP_HEADER = [
    "time_ps", "structure", "coil", "b_sheet", "b_bridge",
    "bend", "turn", "a_helix", "5_helix", "3_helix",
]


# ==================== 组编号解析 ====================

def _parse_group_table(text: str) -> Dict[str, str]:
    """解析 GROMACS 输出的组列表 → {组名小写: 编号}。

    兼容两种格式:
      1) make_ndx 输出:   "  1 Protein     : 1000 atoms"
      2) 分析工具输出:    "Group     1 (  Protein) has 1000 elements"

    返回: {"system": "0", "protein": "1", "c-alpha": "3", ...}
    """
    groups: Dict[str, str] = {}

    # 格式 1: make_ndx (行首数字 + 组名 + 冒号)
    for m in re.finditer(r'^\s*(\d+)\s+(\S+)\s*:\s*\d+', text, re.MULTILINE):
        groups.setdefault(m.group(2).strip().lower(), m.group(1))

    # 格式 2: 分析工具 (Group  N ( name ))
    for m in re.finditer(r'Group\s+(\d+)\s*\(\s*([^)]*?)\s*\)', text, re.IGNORECASE):
        groups.setdefault(m.group(2).strip().lower(), m.group(1))

    return groups


# ==================== 数值格式化 ====================

def _fmt(v: float) -> str:
    """将浮点数格式化为紧凑字符串 (整数不带小数点, 其余 6 位有效数字)。"""
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


# ==================== xvg → CSV ====================

def _xvg_to_csv(
    src: Path,
    dst: Path,
    header: List[str],
    cols: Optional[List[int]] = None,
) -> int:
    """将 gmx 输出 (用 -xvg none 生成) 转换为带表头的 CSV。

    Args:
        src:    源 xvg 文件 (纯数据列, 无 #/@ 注释)
        dst:    目标 CSV 路径
        header: CSV 表头
        cols:   保留的列索引 (None 表示全部保留)

    Returns:
        写入的数据行数
    """
    rows: List[List[float]] = []
    with src.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if cols is not None:
                vals = [vals[c] for c in cols if c < len(vals)]
            if vals:
                rows.append(vals)

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])

    return len(rows)


# ==================== 分析运行器 ====================

class AnalysisRunner:
    """GROMACS MD 后处理分析运行器。

    读取归档目录 (或 MD_result/md/) 中的 md.tpr/md.xtc/md.edr,
    依次执行 PBC 去除 + 结构量 + 表面·溶剂 + 能量分解,
    全部结果写入 <输入目录>/analysis/ 下的 CSV 文件。
    """

    def __init__(
        self,
        project_root: str | Path,
        stats_dir: Optional[str | Path] = None,
        gmx_cmd: str = "gmx",
        dry_run: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.stats_dir = Path(stats_dir).resolve() if stats_dir else None
        self.gmx_cmd = gmx_cmd
        self.dry_run = dry_run

        # 运行时状态
        self.input_dir: Optional[Path] = None
        self.analysis_dir: Optional[Path] = None
        self.md_tpr: Optional[Path] = None
        self.md_xtc: Optional[Path] = None
        self.md_edr: Optional[Path] = None
        self.no_pbc_xtc: Optional[Path] = None
        self.index_path: Optional[Path] = None
        self.groups: Dict[str, str] = {}

        self.summary: List[str] = []
        self.errors: List[str] = []
        self.files: List[str] = []

    # =====================================================================
    #  内部工具
    # =====================================================================

    def _run_gmx(self, args: List[str], input_text: Optional[str] = None,
                 label: str = "gmx") -> subprocess.CompletedProcess:
        """执行 GROMACS 命令的统一入口 (带 dry-run 支持)。"""
        full_cmd = [self.gmx_cmd] + args
        if self.dry_run:
            in_info = f"  |  stdin: {input_text!r}" if input_text else ""
            print(f"  {Color.YELLOW}[DRY-RUN]{Color.RESET} "
                  f"cd {self.analysis_dir} && {' '.join(full_cmd)}{in_info}")
            return subprocess.CompletedProcess(args=full_cmd, returncode=0)

        try:
            result = run_gmx(
                full_cmd,
                cwd=str(self.analysis_dir) if self.analysis_dir else None,
                input=input_text,
                capture_output=True,
            )
            result.check_returncode()
            return result
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or e.stdout or "").strip().splitlines()
            tail_text = "\n".join(tail[-5:]) if tail else "(无输出)"
            raise RuntimeError(f"{label} 失败: {tail_text}") from e
        except FileNotFoundError:
            raise RuntimeError(f"找不到 GROMACS 命令 '{self.gmx_cmd}'") from None

    def _convert(self, xvg: Path, csv_path: Path, header: List[str],
                 cols: Optional[List[int]] = None) -> int:
        """xvg → CSV, dry-run 模式下跳过 (不读文件)。"""
        if self.dry_run:
            return 0
        return _xvg_to_csv(xvg, csv_path, header, cols)

    def _register(self, name: str, csv_path: Path, nrows: int,
                  note: str = "") -> None:
        """登记一个分析产物到 summary/files。"""
        rel = csv_path.name
        self.files.append(rel)
        msg = f"{name:<18s} → {rel:<22s} ({nrows} 行)"
        if note:
            msg += f"  [{note}]"
        self.summary.append(msg)
        info(f"  {name}: {rel} ({nrows} 行)")

    # =====================================================================
    #  输入定位
    # =====================================================================

    def _locate_inputs(self) -> None:
        """定位包含 md.tpr/md.xtc/md.edr 的输入目录。"""
        candidates: List[Path] = []
        if self.stats_dir:
            candidates.append(self.stats_dir)
        candidates.append(self.project_root / "MD_result" / "md")

        chosen: Optional[Path] = None
        for d in candidates:
            if self.dry_run:
                chosen = d
                break
            if all((d / f).is_file() for f in _REQUIRED_INPUTS):
                chosen = d
                break

        if chosen is None:
            raise RuntimeError(
                "找不到成品模拟输入文件 (md.tpr / md.xtc / md.edr)。\n"
                "请先运行 05_md.py 完成成品模拟, 或确认归档目录正确。"
            )

        self.input_dir = chosen
        self.md_tpr = chosen / "md.tpr"
        self.md_xtc = chosen / "md.xtc"
        self.md_edr = chosen / "md.edr"
        self.analysis_dir = chosen / "analysis"

    # =====================================================================
    #  索引文件 + 组编号
    # =====================================================================

    def _generate_index(self) -> None:
        """生成默认 index.ndx 并动态解析 Protein/C-alpha/System 组编号。"""
        self.index_path = self.analysis_dir / "index.ndx"
        groups: Dict[str, str] = {}

        if not self.dry_run:
            try:
                result = run_gmx(
                    [self.gmx_cmd, "make_ndx",
                     "-f", str(self.md_tpr), "-o", str(self.index_path)],
                    cwd=str(self.analysis_dir),
                    input="q\n",
                    capture_output=True,
                )
                if result.returncode == 0:
                    groups = _parse_group_table(result.stdout or "")
            except (subprocess.CalledProcessError, FileNotFoundError):
                groups = {}

        if groups:
            self.groups = groups
        else:
            warn("未能解析组列表, 回退到默认编号 (System=0, Protein=1, C-alpha=3)")
            self.groups = {"system": "0", "protein": "1",
                           "c-alpha": "3", "backbone": "4"}

        # 确保关键组存在
        self.groups.setdefault("system", "0")
        self.groups.setdefault("protein", "1")
        self.groups.setdefault("c-alpha", "3")

    # =====================================================================
    #  PBC 去除
    # =====================================================================

    def _remove_pbc(self) -> None:
        """用 trjconv 去除周期边界 (分子居中 + 整分子 wrap)。"""
        self.no_pbc_xtc = self.analysis_dir / "md_noPBC.xtc"
        protein = self.groups.get("protein", "1")
        system = self.groups.get("system", "0")

        self._run_gmx(
            ["trjconv", "-s", str(self.md_tpr), "-f", str(self.md_xtc),
             "-o", str(self.no_pbc_xtc), "-pbc", "mol", "-center"],
            input_text=f"{protein}\n{system}\n",
            label="trjconv (去除 PBC)",
        )
        info(f"  PBC 去除完成 → {self.no_pbc_xtc.name}")

    # =====================================================================
    #  结构量分析
    # =====================================================================

    def _rmsd(self) -> None:
        ca = self.groups.get("c-alpha", "3")
        xvg = self.analysis_dir / "rmsd.xvg"
        csv_path = self.analysis_dir / "rmsd.csv"
        self._run_gmx(
            ["rms", "-s", str(self.md_tpr), "-f", str(self.no_pbc_xtc),
             "-o", str(xvg), "-tu", "ns", "-xvg", "none"],
            input_text=f"{ca}\n{ca}\n",
            label="rms (RMSD)",
        )
        n = self._convert(xvg, csv_path, ["time_ns", "rmsd_nm"], cols=[0, 1])
        note = ""
        if n:
            vals = self._read_csv_col(csv_path, 1)
            if vals:
                note = f"均值 {sum(vals) / len(vals):.3f} nm, 末值 {vals[-1]:.3f} nm"
        self._register("RMSD", csv_path, n, note)

    def _gyrate(self) -> None:
        protein = self.groups.get("protein", "1")
        xvg = self.analysis_dir / "gyrate.xvg"
        csv_path = self.analysis_dir / "gyrate.csv"
        self._run_gmx(
            ["gyrate", "-s", str(self.md_tpr), "-f", str(self.no_pbc_xtc),
             "-o", str(xvg), "-xvg", "none"],
            input_text=f"{protein}\n",
            label="gyrate (回转半径)",
        )
        n = self._convert(xvg, csv_path, ["time_ps", "rg_nm"], cols=[0, 1])
        note = ""
        if n:
            vals = self._read_csv_col(csv_path, 1)
            if vals:
                note = f"均值 {sum(vals) / len(vals):.3f} nm, 末值 {vals[-1]:.3f} nm"
        self._register("回转半径 Rg", csv_path, n, note)

    def _rmsf(self) -> None:
        ca = self.groups.get("c-alpha", "3")
        xvg = self.analysis_dir / "rmsf.xvg"
        csv_path = self.analysis_dir / "rmsf.csv"
        self._run_gmx(
            ["rmsf", "-s", str(self.md_tpr), "-f", str(self.no_pbc_xtc),
             "-o", str(xvg), "-res", "-xvg", "none"],
            input_text=f"{ca}\n",
            label="rmsf (RMSF)",
        )
        n = self._convert(xvg, csv_path, ["resid", "rmsf_nm"], cols=[0, 1])
        note = ""
        if n:
            vals = self._read_csv_col(csv_path, 1)
            if vals:
                note = f"最大值 {max(vals):.3f} nm"
        self._register("RMSF", csv_path, n, note)

    def _sasa(self) -> None:
        protein = self.groups.get("protein", "1")
        xvg = self.analysis_dir / "sasa.xvg"
        residue_xvg = self.analysis_dir / "residue_sasa.xvg"
        csv_path = self.analysis_dir / "sasa.csv"
        residue_csv = self.analysis_dir / "residue_sasa.csv"
        self._run_gmx(
            ["sasa", "-s", str(self.md_tpr), "-f", str(self.no_pbc_xtc),
             "-o", str(xvg), "-or", str(residue_xvg), "-xvg", "none"],
            input_text=f"{protein}\n",
            label="sasa (溶剂可及表面积)",
        )
        n = self._convert(xvg, csv_path, ["time_ps", "area_nm2"], cols=[0, 1])
        note = ""
        if n:
            vals = self._read_csv_col(csv_path, 1)
            if vals:
                note = f"均值 {sum(vals) / len(vals):.1f} nm²"
        self._register("SASA", csv_path, n, note)

        residue_n = self._convert(
            residue_xvg,
            residue_csv,
            ["resid", "sasa_nm2"],
            cols=[0, 1],
        )
        self._register(
            "SASA（逐残基）",
            residue_csv,
            residue_n,
            "用于候选突变位置推荐",
        )

    def _dssp(self) -> None:
        """二级结构分析 (可选, 依赖外部 DSSP 程序)。"""
        protein = self.groups.get("protein", "1")
        xvg = self.analysis_dir / "scount.xvg"
        csv_path = self.analysis_dir / "ss_count.csv"
        try:
            self._run_gmx(
                ["do_dssp", "-s", str(self.md_tpr), "-f", str(self.no_pbc_xtc),
                 "-sc", str(xvg), "-xvg", "none"],
                input_text=f"{protein}\n",
                label="do_dssp (二级结构)",
            )
        except RuntimeError as e:
            warn(f"  二级结构分析跳过 (通常因未安装 DSSP): {e}")
            self.summary.append("二级结构      → 跳过 (未检测到 DSSP 可执行文件)")
            return

        # 列数随版本略有差异, 按实际列数截断写入
        n = self._convert(xvg, csv_path, _DSSP_HEADER)
        self._register("二级结构", csv_path, n, "需要 DSSP")

    # =====================================================================
    #  能量分解
    # =====================================================================

    def _extract_energy(self, term: str) -> Tuple[List[float], List[float]]:
        """从 md.edr 提取单个能量项 → (time_list, value_list)。"""
        tmp = self.analysis_dir / "_energy_tmp.xvg"
        try:
            self._run_gmx(
                ["energy", "-f", str(self.md_edr), "-o", str(tmp), "-xvg", "none"],
                input_text=f"{term}\n0\n",
                label=f"energy ({term})",
            )
        except RuntimeError:
            return [], []

        if not tmp.is_file():
            return [], []

        times: List[float] = []
        values: List[float] = []
        for line in tmp.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue

        try:
            tmp.unlink()
        except OSError:
            pass

        return times, values

    def _energy(self) -> None:
        """提取各能量项, 合并为一个 energy_components.csv。"""
        data: Dict[str, List[float]] = {}
        time_col: List[float] = []
        found_terms: List[str] = []

        for col, names in _ENERGY_TERMS:
            for name in names:
                t, v = self._extract_energy(name)
                if v:
                    if not time_col:
                        time_col = t
                    data[col] = v
                    found_terms.append(col)
                    break

        if not data:
            warn("  未能从 md.edr 提取任何能量项")
            self.summary.append("能量分解      → 未提取到能量项")
            return

        # 对齐长度 (所有项共享同一 edr 输出频率, 长度应一致, 取最小防越界)
        n = min(len(time_col), *(len(v) for v in data.values()))
        header = ["time_ps"] + list(data.keys())

        csv_path = self.analysis_dir / "energy_components.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(n):
                row = [_fmt(time_col[i])] + [_fmt(data[c][i]) for c in header[1:]]
                writer.writerow(row)

        note = f"{len(found_terms)} 项: {', '.join(found_terms)}"
        self._register("能量分解", csv_path, n, note)

    # =====================================================================
    #  辅助
    # =====================================================================

    @staticmethod
    def _read_csv_col(csv_path: Path, idx: int) -> List[float]:
        """读取 CSV 的第 idx 列 (跳过表头)。"""
        vals: List[float] = []
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            for row in reader:
                if len(row) > idx:
                    try:
                        vals.append(float(row[idx]))
                    except ValueError:
                        continue
        return vals

    def _write_summary(self) -> None:
        """生成 analysis/summary.txt。"""
        if not self.analysis_dir:
            return
        lines: List[str] = []
        sep = "=" * 70
        lines.append(sep)
        lines.append("GROMACS MD 后处理分析报告 (Step 7)")
        lines.append(sep)
        lines.append(f"  输入目录:   {self.input_dir}")
        lines.append(f"  分析目录:   {self.analysis_dir}")
        lines.append("")
        lines.append("分析产物 (CSV):")
        for s in self.summary:
            lines.append(f"  {s}")
        if self.errors:
            lines.append("")
            lines.append("错误:")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append("")
        lines.append(sep)
        lines.append("报告结束")
        lines.append(sep)

        summary_path = self.analysis_dir / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        self.files.append("summary.txt")

    def _result(self) -> dict:
        core_ok = all(
            (self.analysis_dir / f).is_file()
            for f in ("rmsd.csv", "gyrate.csv", "rmsf.csv",
                      "sasa.csv", "residue_sasa.csv",
                      "energy_components.csv")
        ) if self.analysis_dir and not self.dry_run else self.dry_run
        return {
            "success": core_ok and len(self.errors) == 0,
            "analysis_dir": str(self.analysis_dir) if self.analysis_dir else "",
            "input_dir": str(self.input_dir) if self.input_dir else "",
            "summary": self.summary,
            "errors": self.errors,
            "files": self.files,
        }

    # =====================================================================
    #  主流程
    # =====================================================================

    def run(self) -> dict:
        banner("GROMACS MD 后处理分析 (Step 7)")

        try:
            self._locate_inputs()
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            info(f"输入目录: {self.input_dir}")
            info(f"分析目录: {self.analysis_dir}")

            self._generate_index()
            self._remove_pbc()

            # ---- 结构量 + 表面·溶剂 (非致命) ----
            if self.dry_run or self.no_pbc_xtc.is_file():
                for fn in (self._rmsd, self._gyrate, self._rmsf,
                           self._sasa, self._dssp):
                    try:
                        fn()
                    except RuntimeError as e:
                        self.errors.append(str(e))
                        error(str(e))
            else:
                self.errors.append("PBC 去除未成功, 跳过结构分析")
                error("PBC 去除未成功, 跳过结构分析")

            # ---- 能量分解 (非致命) ----
            try:
                self._energy()
            except RuntimeError as e:
                self.errors.append(str(e))
                error(str(e))

        except Exception as e:
            self.errors.append(str(e))
            error(str(e))

        try:
            self._write_summary()
        except Exception as e:
            warn(f"写入 summary.txt 失败: {e}")

        return self._result()


# =========================================================================
#  公共 API — 供 run_all_md.py 通过 import 调用
# =========================================================================

def run_analysis(
    project_root: str | Path,
    stats_dir: Optional[str | Path] = None,
    gmx_cmd: str = "gmx",
    dry_run: bool = False,
) -> dict:
    """执行 GROMACS MD 后处理分析, 输出 CSV 表格。

    Args:
        project_root: 项目根目录
        stats_dir:    归档目录 (output_statistics/<batch>), 找不到时回退 MD_result/md/
        gmx_cmd:      GROMACS 命令, 默认 "gmx"
        dry_run:      只打印命令不执行

    Returns:
        dict:
            success      — bool   核心分析是否全部成功
            analysis_dir — str    分析产物目录
            input_dir    — str    实际输入目录
            summary      — [str]  逐项分析摘要
            errors       — [str]  非致命错误列表
            files        — [str]  生成的 CSV 文件名
    """
    runner = AnalysisRunner(
        project_root=project_root,
        stats_dir=stats_dir,
        gmx_cmd=gmx_cmd,
        dry_run=dry_run,
    )
    return runner.run()


# =========================================================================
#  CLI 入口 — 作为独立脚本运行
# =========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="07_analysis.py",
        description="GROMACS MD 后处理分析 — 结构量 / 表面·溶剂 / 能量分解 → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/07_analysis.py --stats-dir output_statistics/20260827_120000
  python scripts/07_analysis.py --dry-run
        """,
    )
    parser.add_argument(
        "--project-root", default=".",
        help="项目根目录 (默认: 当前工作目录)",
    )
    parser.add_argument(
        "--stats-dir", default=None,
        help="归档目录 output_statistics/<batch> (默认: 自动回退 MD_result/md/)",
    )
    parser.add_argument(
        "--gmx", default="gmx",
        help="GROMACS 命令 (默认: gmx)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式 — 只打印命令不执行",
    )
    args = parser.parse_args(argv)

    result = run_analysis(
        project_root=args.project_root,
        stats_dir=args.stats_dir,
        gmx_cmd=args.gmx,
        dry_run=args.dry_run,
    )

    if result["success"]:
        print(f"\n{'=' * 60}")
        print(f"  分析完成")
        print(f"  产物目录: {result['analysis_dir']}")
        for s in result["summary"]:
            print(f"    {s}")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"  分析部分失败")
        for e in result.get("errors", []):
            print(f"  [错误] {e}")
        print(f"  产物目录: {result['analysis_dir']}")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
