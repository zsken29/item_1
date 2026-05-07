# items_1 项目 AGENTS.md

本文件为 OpenCode 提供项目级上下文和行为约定。

## 目录结构

```
C:\codes\python\items_1\
├── VLMCompression/      # VLM 压缩研究（主要项目）
│   └── code/            # LLaVA/Bunny/InternVL pruning 代码
├── CSTVR/               # 视频恢复项目
│   └── src/             # 源码（arch/data/eval/metrics/module/test/utils）
├── CSTVR_ori/           # 原始 CSTVR 备份
├── lop_repro/           # LOP 复现项目
├── ai_test_1/           # AI 测试项目
│   └── mllm/            # 多模态大模型实验（当前工作目录）
├── ai-test_1/           # AI 测试项目（备选）
├── practice/            # 练习代码
│   ├── decorator_practice/
│   ├── io_practice/
│   ├── image_practice/
│   └── algorithm_practice/
├── read_paper/          # 论文阅读（31篇 VLM 压缩相关论文）
├── 人工智能课程/          # 课程资料
├── 论文下载整理/          # 论文整理
├── .claude/             # 旧 Claude Code 配置（保留）
├── .opencode/           # OpenCode 插件配置
├── minimax_demo.py      # MiniMax API 演示脚本
├── test_image.png       # 测试图片
└── generated_music.mp3  # 生成的音乐
```

## 主要项目说明

### VLMCompression
- VLM 压缩研究：LLaVA/Bunny/InternVL 结构化剪枝
- 代码位于 `code/`，包含 VLM/LLM-Pruner/ShortGPT 等子模块
- 使用 `vlmc` conda 环境

### CSTVR
- 视频恢复（Video Restoration）
- 架构：arch/data/eval/metrics/module/test/utils
- 使用 `cv_env` conda 环境

## 通用规则

- 用中文回复
- 回复简洁，最终回复说明完成内容和改动文件
- 引用文件使用绝对路径，引用行号使用 `file_path:line_number`
- 修改完成后运行相关验证
- 主动管理 git，不提交模型权重/数据集/缓存

## 环境配置

| 项目 | Conda 环境 | Python 路径 |
|------|-----------|------------|
| VLMCompression | vlmc | `conda run -n vlmc python` |
| CSTVR / lop_repro | cv_env | `C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe` |
| 通用 | d2l_env | `conda run -n d2l_env python` |

## MiniMax API

- 端点: `https://api.minimaxi.com`
- 已验证: 搜索、VLM、音乐生成
- MCP 配置位于 `.claude/settings.local.json`

## 验证命令

- Python: `conda run -n <env> python <script>`
- Git: `git status` / `git diff`

## Trae IDE 优化配置

### 配置文件位置
- **Trae 全局设置**: `C:\Users\ZSQ\.trae\settings.json`
- **项目规则**: `c:\codes\python\items_1\.trae\rules\project_rules.md`
- **代码风格**: `c:\codes\python\items_1\.trae\rules\python_style.md`
- **AI 助手指南**: `c:\codes\python\items_1\.trae\ai_context\instructions.md`

### 快速启动
```powershell
# 环境切换
cd c:\codes\python\items_1
.\switch_env.ps1 list                    # 列出环境
.\switch_env.ps1 activate vlmc          # 激活环境
conda run -n vlmc python script.py      # 运行脚本
```

### Git 快捷命令
```bash
gst      # git status
gaa      # git add .
gcm msg  # git commit -m msg
gps      # git push
gpl      # git pull
glg      # git log --graph
```

### 代码片段
- Python 片段: `.trae\snippets\python.json`
- Markdown 片段: `.trae\snippets\markdown.json`
- 使用: `Ctrl + Shift + P` → "Configure User Snippets"

### 项目模板
- ML 模板: `.trae\templates\python_ml\`
- 包含: model.py, data.py, trainer.py, train.py

### 完整文档
- **快速开始**: `c:\codes\python\items_1\VIBE_CODING_SETUP.md`
- **快速参考**: `c:\codes\python\items_1\.trae\QUICK_REFERENCE.md`
- **文件清单**: `c:\codes\python\items_1\.trae\FILES_INDEX.md`
