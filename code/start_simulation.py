#!/usr/bin/env python3
r"""
APEH v2.11 温度启动脚本 — 指定模拟温度后跑全流程, 结束后自动还原 mdp

背景
----
mdp_files/ 里的默认协议是 300 K / 1 ns 的冒烟测试, 不能支撑高温(如 90 °C)结论。
本脚本让操作者在**不改动原始 mdp 文件**的前提下指定模拟温度跑一次全流程。

它做四件事
----------
1. 把选定温度写入 nvt.mdp / npt.mdp / md.mdp 的 `ref-t`
   (nvt.mdp 另含 `gen-temp`, 用于生成初始速度);
2. 原始 mdp 备份到 mdp_runs/<时间戳>/original/, 本次生效的存档到 applied/
   (含 SHA256, 便于复现与审计);
3. 调用 run_all.py 跑全流程, 其余参数原样转发;
4. **全流程结束后 (成功/失败/中断都一样) 把 mdp_files/ 还原为原始参数**,
   并逐文件校验 SHA256 确认逐字节还原。

用法 (解释器用 conda 环境 apeh, 不要用裸 python)
----
  python start_simulation.py                 # 交互式询问温度
  python start_simulation.py --celsius 90    # 90 °C
  python start_simulation.py --temp 363.15   # 363.15 K
  python start_simulation.py --celsius 90 --md-ns 10 --num-sequences 50
  python start_simulation.py --dry-run       # 只预览, 不真正跑 gmx

注意
----
* 还原是**强制**的; 若想取用本次温度下的 mdp, 从 applied/ 目录复制即可。
* 高温提示: 本流程用水模型在近 298 K 参数化, 温度越高结果越需谨慎解读;
  脚本对 > 350 K 给出警告 (不阻断)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_all  # noqa: E402  复用其自适应路径解析


# =========================================================================
#  路径与常量
# =========================================================================

PROJECT_ROOT = run_all.GROMACS_DIR            # MD 工作目录
MDP_DIR = PROJECT_ROOT / "mdp_files"          # 参数模板目录
RUNS_DIR = PROJECT_ROOT / "mdp_runs"          # 备份 / 存档目录
RUN_ALL = HERE / "run_all.py"

# 运行中标记: 存在即表示"上次改写后没能还原", 下次启动会自动补还原
ACTIVE_MARKER = RUNS_DIR / "_active.json"

TEMP_MDP_FILES = ("nvt.mdp", "npt.mdp", "md.mdp")
MD_MDP = "md.mdp"

# ref-t / gen-temp 行: 保留原有缩进、键名大小写与行尾注释, 只替换数值
_TEMP_LINE_RE = re.compile(
    r"^(?P<lead>\s*(?P<key>ref[-_]?t|gen[-_]?temp)\s*=\s*)"
    r"(?P<val>[^;\s]+(?:\s+[^;\s]+)*)"
    r"(?P<tail>\s*(?:;.*)?)$",
    re.IGNORECASE,
)

_NSTEPS_LINE_RE = re.compile(
    r"^(?P<lead>\s*nsteps\s*=\s*)(?P<val>\d+)(?P<tail>\s*(?:;.*)?)$",
    re.IGNORECASE,
)

_DT_LINE_RE = re.compile(
    r"^\s*dt\s*=\s*(?P<val>[\d.eE+-]+)\s*(?:;.*)?$",
    re.IGNORECASE,
)


# =========================================================================
#  基础工具
# =========================================================================

def _emit(text: str, stream) -> None:
    """打印但绝不因输出故障（如管道被关闭）而抛出——否则会中断 mdp 还原。"""
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001
        pass


def info(msg: str) -> None:
    _emit(f"[INFO]  {msg}", sys.stdout)


def warn(msg: str) -> None:
    _emit(f"[WARN]  {msg}", sys.stdout)


def error(msg: str) -> None:
    _emit(f"[ERROR] {msg}", sys.stderr)


def heading(text: str = "") -> None:
    _emit(text, sys.stdout)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt_temp(value: float) -> str:
    """363.15 -> '363.15'; 300.0 -> '300'。"""
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def k_to_c(kelvin: float) -> float:
    return kelvin - 273.15


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


# =========================================================================
#  备份 / 还原
# =========================================================================

def restore_files(backup_dir: Path, expected: dict[str, str]) -> tuple[bool, list[str]]:
    """把 backup_dir 里的文件复制回 mdp_files/ 并校验 SHA256。

    此函数**不打印**, 以便在输出已损坏时仍能完成还原。
    """
    ok = True
    problems: list[str] = []
    for name, digest in expected.items():
        source = backup_dir / name
        destination = MDP_DIR / name
        try:
            if not source.is_file():
                ok = False
                problems.append(f"{name}: 备份缺失 ({source})")
                continue
            shutil.copy2(source, destination)
            if sha256_of(destination) != digest:
                ok = False
                problems.append(f"{name}: 还原后 SHA256 不一致")
        except Exception as exc:  # noqa: BLE001
            ok = False
            problems.append(f"{name}: {exc}")
    return ok, problems


def recover_previous_run() -> bool:
    """若上次运行异常终止（含被强杀），用其备份补还原。返回是否发生了补还原。"""
    if not ACTIVE_MARKER.is_file():
        return False
    try:
        data = json.loads(ACTIVE_MARKER.read_text(encoding="utf-8"))
        backup_dir = Path(data["original_dir"])
        originals = dict(data["originals"])
    except Exception as exc:  # noqa: BLE001
        warn(f"发现残留的运行标记但无法解析（{exc}），已忽略: {ACTIVE_MARKER}")
        ACTIVE_MARKER.unlink(missing_ok=True)
        return False

    warn("检测到上次运行没有正常还原 mdp（可能被强制中断），正在补还原……")
    ok, problems = restore_files(backup_dir, originals)
    for problem in problems:
        error(f"  {problem}")
    if ok:
        info("  补还原完成")
        ACTIVE_MARKER.unlink(missing_ok=True)
    else:
        error(f"  补还原未完全成功, 备份仍在: {backup_dir}")
    return True


# =========================================================================
#  mdp 改写
# =========================================================================

def set_temperature(text: str, temp_k: float) -> tuple[str, list[str]]:
    """把 ref-t / gen-temp 的数值替换为 temp_k，保留原有数值个数与行尾注释。"""
    new_lines: list[str] = []
    replaced: list[str] = []
    for line in text.splitlines():
        match = _TEMP_LINE_RE.match(line)
        if not match:
            new_lines.append(line)
            continue
        tokens = match.group("val").split()
        numbers = [t for t in tokens if re.fullmatch(r"[\d.]+", t)]
        if not numbers:
            new_lines.append(line)
            continue
        new_tokens = [
            fmt_temp(temp_k) if re.fullmatch(r"[\d.]+", t) else t for t in tokens
        ]
        new_lines.append(
            f"{match.group('lead')}{' '.join(new_tokens)}{match.group('tail')}"
        )
        replaced.append(f"{match.group('key')} ({len(numbers)} 值)")
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + trailing, replaced


def read_dt_ps(text: str) -> float:
    """从 md.mdp 读 dt (ps)，读不到按 0.002。"""
    for line in text.splitlines():
        match = _DT_LINE_RE.match(line)
        if match:
            try:
                return float(match.group("val"))
            except ValueError:
                break
    return 0.002


def set_md_nsteps(text: str, target_ns: float) -> tuple[str, int]:
    """按目标时长 (ns) 重算 md.mdp 的 nsteps。返回 (新文本, 新 nsteps)。"""
    dt_ps = read_dt_ps(text)
    nsteps = int(round(target_ns * 1000.0 / dt_ps))
    new_lines: list[str] = []
    hit = False
    for line in text.splitlines():
        match = _NSTEPS_LINE_RE.match(line)
        if match and not hit:
            new_lines.append(f"{match.group('lead')}{nsteps}{match.group('tail')}")
            hit = True
        else:
            new_lines.append(line)
    if not hit:
        raise RuntimeError("md.mdp 中未找到 nsteps 行, 无法按时长改写")
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + trailing, nsteps


# =========================================================================
#  交互与参数
# =========================================================================

def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="start_simulation.py",
        description="指定模拟温度运行 v2.11 全流程, 结束后自动还原 mdp_files/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "其余未列出的参数会原样转发给 run_all.py, 例如:\n"
            "  --gmx gmx --num-sequences 50 --no-gpu --dry-run\n"
            "  --existing-stats-dir output_statistics/<批次>\n"
            '  --design-positions "A35 A68 A102" --omit-aas PC\n'
        ),
    )
    parser.add_argument("--temp", type=float, help="模拟温度, 单位 K (如 363.15)")
    parser.add_argument("--celsius", type=float, help="模拟温度, 单位 °C (如 90)")
    parser.add_argument("--md-ns", type=float, default=None,
                        help="可选: 生产 MD 时长 (ns)。不填则保持 md.mdp 原值")
    args, forwarded = parser.parse_known_args(argv)
    return args, forwarded


def _prompt_temperature() -> float:
    print()
    print("请输入本次模拟的目标温度：")
    print("  * 直接填数字    -> 单位 K（绝对温度），例如 363.15")
    print("  * 数字后加 C    -> 单位 °C，例如 90C")
    sys.stdout.flush()
    while True:
        try:
            raw = input("温度 > ").strip()
        except EOFError:
            raise SystemExit("错误: 没有可用输入，请用 --celsius 或 --temp 指定温度")
        if not raw:
            print("  不能为空, 请重试（例如 90C 或 363.15）", flush=True)
            continue
        if raw.upper().endswith("C"):
            body = raw[:-1].strip().strip("°").strip()
            try:
                return c_to_k(float(body))
            except ValueError:
                print("  无法解析, 请重试（例如 90C 或 363.15）", flush=True)
                continue
        try:
            return float(raw)
        except ValueError:
            print("  无法解析, 请重试（例如 90C 或 363.15）", flush=True)


def resolve_temperature(args: argparse.Namespace) -> float:
    if args.temp is not None and args.celsius is not None:
        raise SystemExit("错误: --temp 与 --celsius 不能同时使用")
    if args.temp is not None:
        temp_k = args.temp
    elif args.celsius is not None:
        temp_k = c_to_k(args.celsius)
    else:
        temp_k = _prompt_temperature()

    if temp_k <= 0:
        raise SystemExit(f"错误: 温度必须为正数 (收到 {temp_k} K)")
    if temp_k < 250 or temp_k > 500:
        warn(f"温度 {fmt_temp(temp_k)} K ({fmt_temp(k_to_c(temp_k))} °C) "
             "超出常见力场验证区间 (250~500 K), 结果请谨慎解读")
    elif temp_k > 350:
        warn(f"高温提醒: {fmt_temp(temp_k)} K ({fmt_temp(k_to_c(temp_k))} °C)。"
             "本流程的显式水模型在近 298 K 参数化, 高温下水的性质偏差会放大, "
             "用其结果支撑高温结论时需注明该局限")
    return temp_k


# =========================================================================
#  主流程
# =========================================================================

def main(argv: list[str]) -> int:
    args, forwarded = parse_args(argv)

    if not MDP_DIR.is_dir():
        error(f"找不到 mdp 目录: {MDP_DIR}")
        return 2
    if not RUN_ALL.is_file():
        error(f"找不到主控脚本: {RUN_ALL}")
        return 2

    # ---- 0. 上次异常终止（含被强杀）的兜底还原 ----
    recover_previous_run()

    temp_k = resolve_temperature(args)
    heading()
    heading("=" * 70)
    info(f"目标温度:   {fmt_temp(temp_k)} K  ({fmt_temp(k_to_c(temp_k))} °C)")
    info(f"MDP 目录:   {MDP_DIR}")
    if args.md_ns is not None:
        info(f"生产 MD:    {fmt_temp(args.md_ns)} ns（覆盖默认值）")
    heading("=" * 70)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / stamp
    orig_dir = run_dir / "original"
    applied_dir = run_dir / "applied"
    orig_dir.mkdir(parents=True, exist_ok=True)
    applied_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 备份原始 mdp（整目录）----
    originals: dict[str, str] = {}
    for source in sorted(MDP_DIR.iterdir()):
        if not source.is_file():
            continue
        shutil.copy2(source, orig_dir / source.name)
        originals[source.name] = sha256_of(source)
    info(f"已备份原始 mdp -> {orig_dir}")

    manifest: dict[str, object] = {
        "timestamp": stamp,
        "temperature_K": temp_k,
        "temperature_C": k_to_c(temp_k),
        "md_ns_override": args.md_ns,
        "mdp_files_temperature_applied": list(TEMP_MDP_FILES),
        "run_all_forwarded_args": forwarded,
        "original_sha256": originals,
    }

    # ---- 2. 写运行标记: 存在即代表"已改写、尚未还原" ----
    ACTIVE_MARKER.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "run_dir": str(run_dir),
                "original_dir": str(orig_dir),
                "originals": originals,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code: int | None = None
    restore_ok = False
    restore_problems: list[str] = []
    try:
        # ---- 3. 改写 mdp ----
        applied_sha: dict[str, str] = {}
        for name in TEMP_MDP_FILES:
            path = MDP_DIR / name
            if not path.is_file():
                raise RuntimeError(f"缺少 mdp 文件: {path}")
            text = path.read_text(encoding="utf-8")
            new_text, replaced = set_temperature(text, temp_k)
            if not replaced:
                raise RuntimeError(
                    f"{name} 中未找到 ref-t / gen-temp, 无法写入温度; 请检查模板"
                )
            if name == MD_MDP and args.md_ns is not None:
                new_text, nsteps = set_md_nsteps(new_text, args.md_ns)
                manifest["md_nsteps"] = nsteps
                info(f"  {name}: {' / '.join(replaced)} -> {fmt_temp(temp_k)}; "
                     f"nsteps -> {nsteps}")
            else:
                info(f"  {name}: {' / '.join(replaced)} -> {fmt_temp(temp_k)}")
            path.write_text(new_text, encoding="utf-8")
            shutil.copy2(path, applied_dir / name)
            applied_sha[name] = sha256_of(path)

        manifest["applied_sha256"] = applied_sha
        info(f"本次生效的 mdp 已存档 -> {applied_dir}")

        # ---- 4. 跑全流程（其余参数原样转发）----
        command = [sys.executable, str(RUN_ALL), *forwarded]
        manifest["command"] = command
        heading()
        info("开始运行全流程: " + " ".join(command))
        heading("=" * 70)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
        exit_code = completed.returncode
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        error(f"{type(exc).__name__}: {exc}")
        exit_code = 2
    finally:
        # ---- 5. 强制还原原始 mdp ----
        # 关键: 先完成文件还原, 之后才打印; 打印失败也不会影响还原结果。
        restore_ok, restore_problems = restore_files(orig_dir, originals)
        if restore_ok:
            try:
                ACTIVE_MARKER.unlink()
            except Exception:  # noqa: BLE001
                pass
        manifest["restore_ok"] = restore_ok
        manifest["exit_code"] = exit_code
        try:
            (run_dir / "run_config.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            error(f"写入 run_config.json 失败: {exc}")

    heading()
    heading("=" * 70)
    if restore_ok:
        info(f"mdp_files/ 已还原为原始参数 "
             f"（{len(originals)} 个文件, SHA256 逐一校验通过）")
    else:
        error("mdp 还原未完全成功：")
        for problem in restore_problems:
            error(f"  {problem}")
        error(f"原始备份仍然保留在: {orig_dir}")
    if exit_code == 0:
        info("全流程结束: 成功")
    else:
        warn(f"全流程结束: 退出码 {exit_code}")
    info(f"本次记录: {run_dir}")
    info("  original/  = 运行前的默认 mdp")
    info("  applied/   = 本次实际生效的 mdp")
    heading("=" * 70)
    return exit_code if exit_code is not None else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
