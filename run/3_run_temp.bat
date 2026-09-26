@echo off
chcp 65001 >nul
title APEH v2.11 - Full Run (choose temperature)
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Full pipeline, choose temperature
echo   You will be asked for a temperature (e.g. 363.15 or 90C).
echo   mdp files are restored to default when the run finishes.
echo ============================================================
echo.

"%PY%" "%CODE%\start_simulation.py" --gmx gmx %*

echo.
pause
