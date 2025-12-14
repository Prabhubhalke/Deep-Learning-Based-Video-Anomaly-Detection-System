@echo off
title Video Anomaly Detection System

echo ========================================
echo Video Anomaly Detection System
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "application.py" (
    echo ERROR: Cannot find application.py
    echo Please run this script from the project directory.
    echo.
    pause
    exit /b 1
)

echo Starting the application...
echo This may take a moment as models are loaded...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
)

REM Run the improved application
python improved_application.py

echo.
echo Application closed.
pause