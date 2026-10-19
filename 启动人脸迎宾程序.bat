@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Face Welcome AI - One Click Start
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SETUP=%ROOT%setup_project.ps1"
set "APP=%ROOT%face_welcome_app.py"

if not exist "%PYTHON%" (
    echo [1/2] Project environment not found. Configuring...
    if not exist "%SETUP%" goto :error_missing_setup
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP%"
    if errorlevel 1 goto :error_setup
)

echo [1/2] Checking startup dependencies...
"%PYTHON%" -c "import tkinter, cv2, insightface, onnxruntime, PIL" >nul 2>&1
if errorlevel 1 (
    echo Dependencies are incomplete. Repairing the environment...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP%"
    if errorlevel 1 goto :error_setup
)

if not exist "%APP%" goto :error_missing_app
echo [2/2] Starting Face Welcome AI...
"%PYTHON%" "%APP%"
if errorlevel 1 goto :error_runtime
exit /b 0

:error_missing_setup
echo.
echo Startup failed: setup_project.ps1 was not found.
goto :pause_error

:error_setup
echo.
echo Startup failed: environment setup did not complete.
goto :pause_error

:error_missing_app
echo.
echo Startup failed: face_welcome_app.py was not found.
goto :pause_error

:error_runtime
echo.
echo Startup failed: the application exited with an error.

:pause_error
echo.
pause
exit /b 1