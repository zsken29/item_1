#!/bin/bash
chmod +x "$0"

echo "========================================"
echo "  中国城市人均数据地图 - 一键启动"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] 检查Python环境..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[错误] 未找到Python，请先安装Python 3.8+"
    read -p "按回车键退出..."
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

echo "[2/4] 安装后端依赖..."
cd "$SCRIPT_DIR/backend"
$PYTHON_CMD -m pip install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo "[错误] 后端依赖安装失败"
    read -p "按回车键退出..."
    exit 1
fi
echo "[OK] 后端依赖安装完成"

echo "[3/4] 初始化数据库..."
$PYTHON_CMD scripts/init_data.py
echo "[OK] 数据库初始化完成"

echo "[4/4] 启动服务..."
echo ""
echo "正在启动后端服务 (http://localhost:8000) ..."
osascript -e 'tell app "Terminal" to do script "cd '"$SCRIPT_DIR/backend"' && '"$PYTHON_CMD"' run.py"' 2>/dev/null || \
("$PYTHON_CMD" run.py &)

sleep 2

echo "正在启动前端服务 (http://localhost:5173) ..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install -q
fi
osascript -e 'tell app "Terminal" to do script "cd '"$SCRIPT_DIR/frontend"' && npm run dev"' 2>/dev/null || \
(npm run dev &)

echo ""
echo "========================================"
echo "  启动完成！"
echo "  后端: http://localhost:8000"
echo "  前端: http://localhost:5173"
echo "========================================"
echo ""

read -p "按回车键打开浏览器..."
open http://localhost:5173 2>/dev/null || xdg-open http://localhost:5173 2>/dev/null || echo "请手动打开浏览器访问 http://localhost:5173"
