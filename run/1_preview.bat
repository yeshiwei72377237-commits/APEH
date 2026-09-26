@echo off
chcp 65001 >nul
title APEH v2.11 - Preview (dry run)
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Preview / dry run
echo   Nothing is executed, no gmx is called.
echo ============================================================
echo.

"%PY%" "%CODE%\run_all.py" --dry-run %*

echo.
pause
