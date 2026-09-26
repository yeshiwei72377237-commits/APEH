#!/usr/bin/env python3
"""
APEH v2.11 环境自检 — 跑全流程前先确认依赖齐全

用法:
  python check_env.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_all  # noqa: E402

PROJECT_ROOT = run_all.GROMACS_DIR

# (显示名, 导入名, 是否必需)
PACKAGES = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("scikit-learn", "sklearn", True),
    ("shap", "shap", True),
    ("aaindex", "aaindex", True),
    ("biopython", "Bio", True),
    ("xgboost（可选）", "xgboost", False),
]

_problems: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}", flush=True)


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}", flush=True)
    _problems.append(msg)


def caution(msg: str) -> None:
    print(f"  [WARN] {msg}", flush=True)
    _warnings.append(msg)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python() -> None:
    section("1. Python 解释器")
    print(f"  解释器: {sys.executable}")
    print(f"  版本:   {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        bad(f"Python 版本过低（{sys.version.split()[0]}），需要 >= 3.9")
    else:
        ok("版本满足要求 (>= 3.9)")


def check_paths() -> None:
    section("2. 工作目录与模型包")
    print(f"  code/        : {HERE}")
    print(f"  工作目录     : {run_all._SRC_CODE_DIR}")
    if run_all._SRC_CODE_DIR.is_dir():
        ok("src_code_dir.txt / 环境变量解析成功")
    else:
        bad(f"工作目录不存在: {run_all._SRC_CODE_DIR}")
    if (HERE / "aaindex_model" / "artifacts" / "final_pipeline.pkl").is_file():
        ok("打分模型包 aaindex_model/artifacts/final_pipeline.pkl 存在")
    else:
        bad("缺少打分模型产物 code/aaindex_model/artifacts/final_pipeline.pkl")


def check_packages() -> None:
    section("3. Python 依赖包")
    for label, module, required in PACKAGES:
        spec = importlib.util.find_spec(module)
        if spec is None:
            if required:
                bad(f"{label} 未安装（import {module} 失败）")
            else:
                caution(f"{label} 未安装（不影响当前模型）")
        else:
            try:
                version = getattr(__import__(module), "__version__", "?")
            except Exception:  # noqa: BLE001
                version = "?"
            ok(f"{label} {version}")


def check_gromacs() -> None:
    section("4. GROMACS")
    gmx = shutil.which("gmx")
    if not gmx:
        bad("PATH 中找不到 gmx 命令；MD 步骤无法运行")
        return
    ok(f"gmx 位置: {gmx}")
    try:
        proc = subprocess.run([gmx, "--version"], capture_output=True, text=True,
                              timeout=60)
        line = next(
            (l.strip() for l in proc.stdout.splitlines() if "GROMACS" in l and "-" in l),
            "",
        )
        ok(f"版本: {line or proc.stdout.splitlines()[0].strip()}")
    except Exception as exc:  # noqa: BLE001
        caution(f"无法获取 gmx 版本: {exc}")


def check_project() -> None:
    section("5. GROMACS 工作目录")
    print(f"  路径: {PROJECT_ROOT}")
    if not PROJECT_ROOT.is_dir():
        bad(f"工作目录不存在: {PROJECT_ROOT}")
        return
    ok("工作目录存在")

    mdp_dir = PROJECT_ROOT / "mdp_files"
    required_mdp = ["em.mdp", "nvt.mdp", "npt.mdp", "md.mdp", "ions.mdp"]
    missing = [f for f in required_mdp if not (mdp_dir / f).is_file()]
    if missing:
        bad(f"缺少 mdp 模板: {missing}")
    else:
        ok(f"mdp 模板齐全（{len(required_mdp)} 个）")

    temps = []
    for name in ("nvt.mdp", "npt.mdp", "md.mdp"):
        path = mdp_dir / name
        if not path.is_file():
            temps.append(f"{name}=?")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?:^|\n)\s*ref-t\s*=\s*([\d.]+)", text)
        temps.append(f"{name}={match.group(1) if match else '?'}")
    print(f"  当前模拟温度: {', '.join(temps)} K")

    pdbs = sorted((PROJECT_ROOT / "input_pdb").glob("*.pdb"))
    if pdbs:
        ok(f"输入 PDB: {', '.join(p.name for p in pdbs)}")
    else:
        caution("input_pdb/ 下没有 .pdb；请放置待改造蛋白或用 --pdb 指定")

    if (PROJECT_ROOT / "scripts").is_dir():
        ok("GROMACS 步骤脚本目录存在")
    else:
        bad(f"缺少步骤脚本目录: {PROJECT_ROOT / 'scripts'}")

    if (PROJECT_ROOT / "mdp_runs" / "_active.json").is_file():
        caution("检测到 mdp_runs/_active.json：上次运行未正常还原，下次启动会自动补还原")


def check_mpnn() -> None:
    section("6. ProteinMPNN")
    base = run_all._SRC_CODE_DIR / "ProteinMPNN" / "ProteinMPNN-main"
    if not (base / "protein_mpnn_run.py").is_file():
        bad(f"找不到主程序: {base / 'protein_mpnn_run.py'}")
        return
    ok("主程序 protein_mpnn_run.py 存在")
    weights = sorted((base / "vanilla_model_weights").glob("*.pt"))
    if weights:
        ok(f"模型权重: {', '.join(w.name for w in weights)}")
    else:
        bad("vanilla_model_weights/ 下没有 .pt 权重文件")
    reason = run_all._validate_proteinmpnn_dir(base)
    if reason:
        bad(f"ProteinMPNN 目录不完整: {reason}")
    else:
        ok("ProteinMPNN 目录完整（主程序 + 权重 + 辅助脚本）")


def check_model() -> None:
    section("7. AAindex 打分模型（Step 10）")
    model_dir = HERE / "aaindex_model"
    artifacts = model_dir / "artifacts"
    for name in ("final_pipeline.pkl", "feature_spec.json"):
        path = artifacts / name
        if path.is_file():
            ok(f"{name}（{path.stat().st_size:,} 字节）")
        else:
            bad(f"缺少模型产物: {path}")
    for name in ("nested_cv.py", "score.py", "run_loop.py", "select_top_mutants.py"):
        if (model_dir / name).is_file():
            ok(f"{name} 存在")
        else:
            bad(f"缺少模型脚本: {model_dir / name}")


def main() -> int:
    print("APEH v2.11 环境自检")
    check_python()
    check_paths()
    check_packages()
    check_gromacs()
    check_project()
    check_mpnn()
    check_model()

    section("汇总")
    if _problems:
        print(f"  发现 {len(_problems)} 个必须解决的问题：")
        for problem in _problems:
            print(f"    - {problem}")
    else:
        print("  关键项全部通过，可以运行全流程。")
    if _warnings:
        print(f"  另有 {len(_warnings)} 条提醒（不阻断）：")
        for warning in _warnings:
            print(f"    - {warning}")
    print()
    return 1 if _problems else 0


if __name__ == "__main__":
    sys.exit(main())
