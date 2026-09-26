@echo off
chcp 65001 >nul
title APEH v2.11 - Restore mdp files
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Restore mdp files to defaults
echo ============================================================
echo.

set "RESTORE=%~1"
if /i "%RESTORE%"=="--restore" goto do_restore

echo Listing available backups ...
echo.
"%PY%" "%CODE%\restore_mdp.py"

echo.
echo ------------------------------------------------------------
set /p ANSWER=Restore mdp files now? [y/N]:
if /i not "%ANSWER%"=="y" goto cancelled

:do_restore
"%PY%" "%CODE%\restore_mdp.py" --restore
echo.
pause
exit /b 0

:cancelled
echo.
echo Cancelled - nothing changed.
echo.
pause
exit /b 0
