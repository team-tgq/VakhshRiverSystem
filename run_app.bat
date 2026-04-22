@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%POWERSHELL_EXE%" (
    echo [ERROR] PowerShell not found: %POWERSHELL_EXE%
    pause
    exit /b 1
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_app.ps1" -PauseOnError
set "APP_EXIT=%ERRORLEVEL%"

exit /b %APP_EXIT%
