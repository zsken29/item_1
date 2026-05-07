# 中国城市人均数据地图 - 一键启动脚本
# 使用方式: 右键 -> 使用PowerShell运行

$ErrorActionPreference = "Continue"
$BASE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND_DIR = Join-Path $BASE_DIR "backend"
$FRONTEND_DIR = Join-Path $BASE_DIR "frontend"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  China City Data Map - One Click Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python..." -NoNewline
try {
    $pythonVersion = python --version 2>&1
    Write-Host " OK ($pythonVersion)" -ForegroundColor Green
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Python is required. Please install Python 3.8+" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install backend deps
Write-Host "[2/5] Installing backend dependencies..." -NoNewline
Set-Location $BACKEND_DIR
pip install -r requirements.txt -q | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Yellow
} else {
    Write-Host " OK" -ForegroundColor Green
}

# Initialize database
Write-Host "[3/5] Initializing database..." -NoNewline
python scripts/init_data.py 2>&1 | Out-Null
Write-Host " OK" -ForegroundColor Green

# Start backend
Write-Host "[4/5] Starting backend server (port 8000)..." -NoNewline
Start-Process python -ArgumentList "run.py" -WindowStyle Normal -WorkingDirectory $BACKEND_DIR
Write-Host " OK" -ForegroundColor Green

Start-Sleep -Seconds 2

# Install frontend deps
Write-Host "[5/5] Starting frontend server (port 5173)..." -NoNewline
Set-Location $FRONTEND_DIR
if (-not (Test-Path "node_modules")) {
    npm install -q 2>&1 | Out-Null
}
Start-Process npm -ArgumentList "run dev" -WindowStyle Normal -WorkingDirectory $FRONTEND_DIR
Write-Host " OK" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All services started!" -ForegroundColor Cyan
Write-Host "  Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Opening browser in 3 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Read-Host "Press Enter to exit"
