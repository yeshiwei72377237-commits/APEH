@echo off
chcp 65001 >nul
title APEH v2.11 - Full Run (default 300 K)
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Full pipeline, default protocol (300 K)
echo   Step 0-11 using the stock mdp files (not modified).
echo ============================================================
echo.

"%PY%" "%CODE%\run_all.py" --gmx gmx %*

echo.
pause
