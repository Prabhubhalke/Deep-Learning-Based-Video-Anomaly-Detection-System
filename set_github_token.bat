@echo off
setlocal

echo ========================================
echo GitHub Token Setup for Video Analysis
echo ========================================

set /p github_token="Enter your GitHub token: "

if "%github_token%"=="" (
    echo Error: GitHub token cannot be empty
    pause
    exit /b 1
)

setx GITHUB_TOKEN "%github_token%"

echo.
echo GitHub token set successfully!
echo You can now run the application with real-time GPT analysis
echo.

pause