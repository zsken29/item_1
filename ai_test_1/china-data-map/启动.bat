@echo off
chcp 65001 >nul
title 中国城市人均数据地图

echo ========================================
echo   中国城市人均数据地图 - 一键启动
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [2/4] 安装后端依赖...
cd backend
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 后端依赖安装失败
    pause
    exit /b 1
)
echo [OK] 后端依赖安装完成

echo [3/4] 初始化数据库...
python scripts/init_data.py
if errorlevel 1 (
    echo [警告] 数据初始化出现问题，继续启动...
)
echo [OK] 数据库初始化完成

echo [4/4] 启动服务...
echo.
echo 正在启动后端服务 (http://localhost:8000) ...
start "Backend" cmd /c "python run.py"

timeout /t 3 /nobreak >nul

echo 正在启动前端服务 (http://localhost:5173) ...
cd ..\frontend
npm install -q
if errorlevel 1 (
    echo [错误] 前端依赖安装失败
    pause
    exit /b 1
)
start "Frontend" cmd /c "npm run dev"

echo.
echo ========================================
echo   启动完成！
echo   后端: http://localhost:8000
echo   前端: http://localhost:5173
echo ========================================
echo.
echo 按任意键打开浏览器...
pause >nul
start http://localhost:5173
