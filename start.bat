@echo off
:: LocalAI Platform — Windows launcher (no terminal window)
cd /d "%~dp0"

:: Use pythonw to run without showing a terminal window
start "" /B pythonw start.py

:: Brief pause then open browser (fallback in case auto-open fails)
timeout /t 8 /nobreak >nul
start http://localhost:3000/api/localai/model-manager
