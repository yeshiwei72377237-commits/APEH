#!/usr/bin/env python3
"""GROMACS subprocess 编码兼容层。

Windows 中文版默认编码为 GBK (cp936)。GROMACS 输出含科学符号
（Å、β、° 等），其字节序列在 GBK 和 UTF-8 之间不兼容，
直接使用 text=True 会因编码推断错误而抛 UnicodeDecodeError。

本模块提供 run_gmx() 作为 subprocess.run 的替换：
- 捕获原始 bytes，不做自动编码推断
- 系统编码 → UTF-8 → latin-1 三级降级解码
- 自动保护 sys.stdout/sys.stderr 防止 print() 崩溃
- 返回值与 subprocess.CompletedProcess 完全兼容
"""

from __future__ import annotations

import locale
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union


# =========================================================================
#  终端编码保护 — 确保 print() 到 stdout/stderr 永不崩溃
# =========================================================================

def _protect_output_streams() -> None:
    """为 sys.stdout / sys.stderr 注册 errors='replace'。

    Windows 终端编码通常为 GBK/cp936，当 print() 输出的字符
    超出该编码范围时会抛 UnicodeEncodeError。将 errors 改为
    'replace' 可确保用 ? 替代无法编码的字符，永不崩溃。
    """
    for stream, name in ((sys.stdout, "stdout"), (sys.stderr, "stderr")):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            # reconfigure 可能不被支持（如被重定向到文件），静默忽略
            pass


_protect_output_streams()


# =========================================================================
#  多级降级解码
# =========================================================================

def _safe_decode(raw: bytes) -> str:
    """系统编码 → UTF-8 → latin-1 三级降级，永不抛 UnicodeDecodeError。

    GROMACS Windows 版输出使用系统区域编码（中文 Windows 为 GBK/cp936），
    因此优先尝试系统编码，其次 UTF-8（用于跨平台/Linux 场景），
    最后 latin-1 作为 universal fallback（每个单字节都合法）。
    """
    # 构建编码优先级列表
    encodings: List[str] = []
    try:
        pref = locale.getpreferredencoding(False)
        if pref:
            encodings.append(pref)
    except Exception:
        pass
    # 追加通用编码，去重
    for enc in ("utf-8", "utf-16", "gbk", "latin-1"):
        if enc.lower() not in (e.lower() for e in encodings):
            encodings.append(enc)

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # 最终兜底：UTF-8 + replace
    return raw.decode("utf-8", errors="replace")


# =========================================================================
#  编码健壮的 subprocess.run 替代
# =========================================================================

def run_gmx(
    cmd: List[str],
    *,
    cwd: Optional[Union[str, Path]] = None,
    input: Optional[str] = None,
    capture_output: bool = False,
    timeout: Optional[float] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """编码健壮的 subprocess.run 替代。

    不依赖 text=True 的 locale 编码推断；捕获原始 bytes，
    用 _safe_decode 降级解码 GROMACS 输出。

    Args:
        cmd: 命令 + 参数列表
        cwd: 工作目录
        input: 通过 stdin 传递的文本（自动编码为 UTF-8 bytes）
        capture_output: True 时捕获 stdout/stderr
        timeout: 超时秒数
        check: True 时非零退出码自动抛 CalledProcessError

    Returns:
        CompletedProcess，其 stdout/stderr 属性为已解码的 str。
    """
    input_bytes = input.encode("utf-8") if input is not None else None

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_bytes,
        capture_output=capture_output,
        timeout=timeout,
    )

    stdout_str = _safe_decode(result.stdout) if result.stdout else ""
    stderr_str = _safe_decode(result.stderr) if result.stderr else ""

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=stdout_str,
            stderr=stderr_str,
        )

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=result.returncode,
        stdout=stdout_str,
        stderr=stderr_str,
    )


# =========================================================================
#  错误输出格式化 — 显示头+尾，避免 GROMACS 横幅淹没真正的错误
# =========================================================================

def format_error_output(stderr_text: str, stdout_text: str,
                        head_chars: int = 200,
                        tail_chars: int = 2000) -> str:
    """将 stderr/stdout 格式化为诊断用文本。

    GROMACS 把版本横幅放在输出开头，把真正的错误信息放在末尾。
    如果只截取前 N 字符，永远看不到错误原因。

    策略：总长度较短 → 全部显示；较长 → 显示前 head_chars + 后 tail_chars。
    """
    combined = (stderr_text + "\n" + stdout_text).strip()
    if not combined:
        return "(无输出)"
    if len(combined) <= head_chars + tail_chars + 500:
        return combined
    head = combined[:head_chars]
    tail = combined[-tail_chars:]
    skipped = len(combined) - head_chars - tail_chars
    sep = f"\n... (中间省略 {skipped} 字符) ...\n"
    return head + sep + tail
