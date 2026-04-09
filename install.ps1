# LocalAI Platform — Windows One-Click Installer
# Run from PowerShell: .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== LocalAI Platform Installer ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "ERROR: Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVersion = python --version 2>&1
Write-Host "Found: $pyVersion"

# Check version >= 3.10
$ver = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$ver -lt [version]"3.10") {
    Write-Host "ERROR: Python 3.10+ required (found $ver)" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
$pip = ".venv\Scripts\pip.exe"
$python = ".venv\Scripts\python.exe"

# Install dependencies
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
& $pip install --upgrade pip -q
& $pip install -r requirements.txt

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start LocalAI Platform:"
Write-Host "  1. Double-click start.bat"
Write-Host "  2. Or run: .venv\Scripts\python.exe start.py"
Write-Host ""
Write-Host "To build LocalAI.exe (optional):"
Write-Host "  & $pip install pyinstaller"
Write-Host "  .\build_exe.ps1"
