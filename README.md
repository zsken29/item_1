# items_1 - 项目总览

## 📁 目录结构

```
items_1/
├── 📂 主要项目
│   ├── VLMCompression/          # VLM 压缩研究 (LLaVA/Bunny/InternVL)
│   ├── CSTVR/                   # 视频恢复项目
│   ├── CSTVR_ori/              # 视频恢复项目备份
│   ├── lop_repro/              # LOP 复现项目
│   ├── ai_test_1/              # AI 测试项目
│   │   ├── mllm/              # 多模态大模型实验
│   │   ├── 迷宫/              # 迷宫游戏
│   │   ├── ai-qa-frontend/    # AI 问答前端
│   │   ├── china-data-map/     # 中国数据地图
│   │   └── hotspot/            # 热点项目
│   └── practice/               # 练习代码
│
├── 📂 文档和配置
│   ├── docs/                    # 文档目录
│   │   ├── AGENTS.md          # AI 助手项目说明
│   │   ├── CLAUDE.md          # Claude 配置
│   │   └── VIBE_CODING_SETUP.md  # Vibe Coding 设置
│   └── scripts/                # 脚本目录
│       ├── quick_commands.sh   # 快速命令 (Linux/Mac)
│       └── switch_env.ps1      # Conda 环境切换 (Windows)
│
├── 📂 学习资源
│   ├── read_paper/            # 论文阅读笔记
│   ├── 人工智能课程/          # 课程资料
│   └── 论文下载整理/          # 论文整理
│
├── 📂 配置文件 (隐藏)
│   ├── .trae/                 # Trae IDE 配置
│   ├── .claude/              # Claude 配置
│   ├── .opencode/            # OpenCode 配置
│   ├── .vscode/              # VS Code 配置
│   └── .git/                 # Git 仓库
│
├── 📄 根目录文件
│   ├── .gitconfig            # Git 全局配置
│   └── .gitignore            # Git 忽略规则
│
└── 📄 本文件
    └── README.md             # 项目总览
```

## 🚀 快速开始

### 环境配置

```powershell
# 切换 Conda 环境
.\scripts\switch_env.ps1 list
.\scripts\switch_env.ps1 activate vlmc
```

### 常用命令

```bash
# Git 操作
gst      # git status
gaa      # git add .
gcm msg  # git commit -m msg
gps      # git push

# 环境管理
conda env list
conda activate <env_name>
conda run -n <env_name> python script.py
```

## 📦 主要项目说明

### VLMCompression
**VLM 压缩研究**
- 位置: `VLMCompression/code/`
- 包含: LLaVA/Bunny/InternVL 剪枝代码
- 环境: `vlmc`

### CSTVR
**视频恢复**
- 位置: `CSTVR/`
- 架构: arch/data/eval/metrics/module/test/utils
- 环境: `cv_env`

### ai_test_1
**AI 测试项目**
- 位置: `ai_test_1/`
- 包含多个子项目
- 环境: 视具体项目而定

### lop_repro
**LOP 复现**
- 位置: `lop_repro/`
- 环境: `cv_env`

### practice
**练习代码**
- 位置: `practice/`
- 包含: 算法、装饰器、图像处理、IO 等练习

## 📚 文档位置

| 文档 | 位置 | 说明 |
|------|------|------|
| AI 助手指南 | [docs/AGENTS.md](docs/AGENTS.md) | 项目上下文和规则 |
| Vibe Coding 设置 | [docs/VIVE_CODING_SETUP.md](docs/VIBE_CODING_SETUP.md) | 开发环境优化 |
| Trae 配置 | `.trae/` | IDE 配置和代码片段 |

## 🔧 环境配置

| 项目 | Conda 环境 | 说明 |
|------|------------|------|
| VLMCompression | `vlmc` | VLM 压缩研究 |
| CSTVR | `cv_env` | 视频恢复 |
| lop_repro | `cv_env` | LOP 复现 |
| 通用 | `d2l_env` | 通用学习 |

## 📖 学习资源

### 论文阅读
- 位置: `read_paper/`
- 包含 31+ 篇 VLM 压缩相关论文笔记

### 课程资料
- 位置: `人工智能课程/`
- 包含课程笔记和代码

## 🎯 开发规范

### Git 提交规范
```
feat: 新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试
chore: 构建/工具
```

### 代码规范
- 遵循 PEP 8
- 使用类型注解
- 添加文档字符串
- 统一命名风格

## 📊 项目统计

- 主要项目: 6 个
- AI 测试子项目: 5 个
- 练习模块: 4 个
- 论文笔记: 31+ 篇

## 🤝 贡献指南

1. 遵循项目规范
2. 提交前检查代码
3. 使用规范的提交信息
4. 不提交敏感信息和大文件

## 📞 联系方式

如有问题，请查看：
- [docs/AGENTS.md](docs/AGENTS.md) - 项目说明
- [docs/VIBE_CODING_SETUP.md](docs/VIBE_CODING_SETUP.md) - 开发设置

---

**最后更新**: 2026-05-07
