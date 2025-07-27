@echo off
REM Data Analysis App Windows Startup Script
REM Version: 1.0.0

REM Application Information
set "APP_NAME=Data Analysis Pro"
set "APP_VERSION=1.0.0"
set "APP_DIR=%~dp0..\"
set "MAIN_SCRIPT=%APP_DIR%main.py"
set "PYTHON_CMD=uv run python"

REM Check if main script exists
if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Main script not found: %MAIN_SCRIPT%
    pause
    exit /b 1
)

REM Start application
echo [INFO] Starting %APP_NAME%...

cd /d "%APP_DIR%"
%PYTHON_CMD% "%MAIN_SCRIPT%" %*