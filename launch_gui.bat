@echo off
title Patch Antenna Simulator - Desktop GUI

echo.
echo ===============================================
echo    🛰️ Patch Antenna Simulator - Desktop GUI
echo ===============================================
echo.

REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo 🐍 Activating virtual environment...
    call .venv\Scripts\activate.bat
    set "VENV_ACTIVATED=1"
) else (
    echo ⚠️  No virtual environment found, checking system Python...
    REM Try py launcher first (Windows Python Launcher)
    py --version >nul 2>&1
    if errorlevel 1 (
        REM Fall back to python command
        python --version >nul 2>&1
        if errorlevel 1 (
            echo ❌ Error: Python is not installed or not in PATH
            echo Please install Python 3.8+ and try again
            echo.
            pause
            exit /b 1
        ) else (
            echo ✅ Using system Python
        )
    ) else (
        echo ✅ Using py launcher
    )
)

echo 🚀 Starting GUI application...
echo.

REM Run the launcher - try py first, then python
if exist ".venv\Scripts\activate.bat" (
    REM Use activated virtual environment
    python launch_gui.py
) else (
    REM Try py launcher first, then fall back to python
    py launch_gui.py 2>nul || python launch_gui.py
)

REM Deactivate virtual environment if it was activated
if defined VENV_ACTIVATED (
    call deactivate 2>nul
)

echo.
echo Application closed.
pause
