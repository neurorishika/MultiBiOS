@echo off
REM MultiBiOS Experiment Explorer launcher
REM Double-click this file to open the explorer in your browser

cd /d "%~dp0"
title MultiBiOS Explorer

echo.
echo  Starting MultiBiOS Experiment Explorer...
echo  It will open automatically in your default browser.
echo  Press Ctrl+C to stop the server.
echo.

"C:\Users\markd\.conda\envs\multibios\python.exe" explorer.py

pause
