#!/bin/bash
# quick-commands.sh
# 常用命令快捷方式

# Python ML 快捷命令
alias pyenv='conda activate'
alias pyenv-list='conda env list'
alias pyenv-create='conda create -n'

# 项目快捷命令
alias cdc='cd c:/codes/python/items_1'
alias cdv='cd c:/codes/python/items_1/VLMCompression'
alias cdcv='cd c:/codes/python/items_1/CSTVR'

# Git 快捷命令
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'
alias gco='git checkout'
alias gb='git branch'
alias gpl='git pull'
alias gmg='git merge'

# 快速查看
alias ll='ls -lah'
alias la='ls -A'
alias l='ls -CF'

# Python 快捷命令
alias py='python'
alias pipi='pip install'
alias pipl='pip list'
alias pyc='python -m py_compile'

# Conda 快捷命令
alias ca='conda activate'
alias cde='conda deactivate'
alias cl='conda env list'

# 运行常用脚本
run-maze() {
    cd "c:/codes/python/items_1/ai_test_1/迷宫"
    conda run -n d2l_env python main.py
}

# 清理命令
clean-pycache() {
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete 2>/dev/null
    echo "清理完成"
}

clean-all() {
    clean-pycache
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
    echo "全面清理完成"
}

# Git 状态
git-status-all() {
    echo "📊 Git 状态概览"
    echo "================"
    for dir in */; do
        if [ -d "$dir/.git" ]; then
            echo -e "\n📁 $dir"
            cd "$dir"
            git status -s | head -5
            cd ..
        fi
    done
}

# 快速备份
backup-file() {
    if [ -f "$1" ]; then
        cp "$1" "$1.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ 已备份: $1"
    else
        echo "❌ 文件不存在: $1"
    fi
}

# 搜索文件
find-py() {
    find . -name "*.py" | grep -i "$1"
}

find-md() {
    find . -name "*.md" | grep -i "$1"
}

# 系统信息
sysinfo() {
    echo "🖥️  系统信息"
    echo "============="
    echo "系统: $(uname -s)"
    echo "用户: $(whoami)"
    echo "时间: $(date)"
    echo ""
    echo "📦 Conda 环境:"
    conda env list | grep -v "^$" | grep -v "^#" | tail -n +2
}

# 帮助信息
show-help() {
    echo "🚀 常用快捷命令"
    echo "================"
    echo ""
    echo "项目导航:"
    echo "  cdc   - 进入 items_1 目录"
    echo "  cdv   - 进入 VLMCompression 目录"
    echo "  cdcv  - 进入 CSTVR 目录"
    echo ""
    echo "Git 命令:"
    echo "  gs    - git status"
    echo "  ga    - git add"
    echo "  gc    - git commit"
    echo "  gp    - git push"
    echo "  gl    - git log"
    echo "  gd    - git diff"
    echo ""
    echo "Python 命令:"
    echo "  py    - 运行 python"
    echo "  pipi  - pip install"
    echo "  pyc   - 编译 Python 文件"
    echo ""
    echo "清理命令:"
    echo "  clean-pycache - 清理 __pycache__"
    echo "  clean-all     - 全面清理"
    echo ""
    echo "实用命令:"
    echo "  sysinfo           - 系统信息"
    echo "  find-py <关键词>  - 搜索 Python 文件"
    echo "  backup-file <文件> - 备份文件"
    echo "  run-maze          - 运行迷宫游戏"
    echo ""
}
