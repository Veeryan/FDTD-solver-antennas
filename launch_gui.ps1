# Patch Antenna Simulator - Desktop GUI Launcher (PowerShell)
# Better virtual environment handling for Windows

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   🛰️ Patch Antenna Simulator - Desktop GUI" -ForegroundColor Cyan  
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Check if virtual environment exists
$venvPath = ".\.venv"
$venvActivate = "$venvPath\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Green
    try {
        & $venvActivate
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Failed to activate virtual environment, using system Python" -ForegroundColor Yellow
    }
}
else {
    Write-Host "⚠️  No virtual environment found at $venvPath" -ForegroundColor Yellow
    Write-Host "💡 Tip: Create one with: python -m venv .venv" -ForegroundColor Blue
}

# Check Python availability
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    }
    else {
        throw "Python not found"
    }
}
catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Starting GUI application..." -ForegroundColor Green
Write-Host ""

try {
    # Run the launcher
    python launch_gui.py
}
catch {
    Write-Host "❌ Error running application: $_" -ForegroundColor Red
}
finally {
    # Deactivate virtual environment if it was activated
    if (Test-Path $venvActivate) {
        deactivate 2>$null
    }
}

Write-Host ""
Write-Host "Application closed." -ForegroundColor Gray
Read-Host "Press Enter to exit"
