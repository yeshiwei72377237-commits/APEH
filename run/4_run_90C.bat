@echo off
chcp 65001 >nul
title APEH v2.11 - Full Run at 90 C
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Full pipeline at 90 C (363.15 K)
echo   mdp files are restored to default when the run finishes.
echo ============================================================
echo.

"%PY%" "%CODE%\start_simulation.py" --celsius 90 --gmx gmx %*

echo.
pause
