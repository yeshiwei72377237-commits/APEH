@echo off
chcp 65001 >nul
title APEH v2.11 - Collect results (Step 11)
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Collect results into result\^<batch\>\
echo ============================================================
echo.

"%PY%" "%CODE%\collect_results.py" --list
echo.
echo ------------------------------------------------------------
echo  Press ENTER to collect the latest batch (or Ctrl+C to stop).
echo ------------------------------------------------------------
pause >nul

"%PY%" "%CODE%\collect_results.py" %*

echo.
pause
