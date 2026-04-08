# LocalAI Platform — Build LocalAI.exe from start.py using PyInstaller
# Run from repo root: .\build_exe.ps1

$ErrorActionPreference = "Stop"

Write-Host "Building LocalAI.exe..." -ForegroundColor Cyan

# Check PyInstaller
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Installing PyInstaller..."
    pip install pyinstaller
}

# Clean previous build
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path build) { Remove-Item build -Recurse -Force }

pyinstaller start.py `
    --onedir `
    --windowed `
    --name "LocalAI" `
    --icon "assets/icon.ico" `
    --add-data "config;config" `
    --add-data "extensions;extensions" `
    --add-data "frontend;frontend" `
    --hidden-import open_webui `
    --hidden-import open_webui.main `
    --hidden-import psutil `
    --hidden-import pynvml `
    --hidden-import gguf `
    --hidden-import pystray `
    --hidden-import PIL `
    --hidden-import uvicorn `
    --hidden-import fastapi `
    --collect-all open_webui `
    --noconfirm

Write-Host ""
Write-Host "Done! LocalAI.exe is in: dist/LocalAI/" -ForegroundColor Green
Write-Host "Double-click dist/LocalAI/LocalAI.exe to launch."
