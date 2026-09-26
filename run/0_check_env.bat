@echo off
chcp 65001 >nul
title APEH v2.11 - Environment Check
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=D:\EvoEngine\envs\apeh\python.exe"
set "CODE=D:\apeh\v2.11\code"

echo ============================================================
echo   APEH v2.11  -  Environment Check
echo ============================================================
echo.

if not exist "%PY%" (
    echo [FAIL] Python not found: %PY%
    echo        Please check the interpreter path.
    echo.
    pause
    exit /b 2
)

"%PY%" "%CODE%\check_env.py"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [DONE] Environment looks good.
) else (
    echo [DONE] Problems found - see the list above.
)
echo.
pause
exit /b %RC%
