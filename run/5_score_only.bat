@echo off
chcp 65001 >nul
title APEH v2.11 - Scoring only (Step 10)
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Score ProteinMPNN candidates (Step 10 only)
echo ============================================================
echo.

set "RUNDIR=%~1"
if "%RUNDIR%"=="" set /p RUNDIR=ProteinMPNN run dir (or drag folder here):

echo.
echo Scoring: %RUNDIR%
echo.

"%PY%" "%CODE%\aaindex_model\run_loop.py" --dir "%RUNDIR%" --top-k 20

echo.
pause
