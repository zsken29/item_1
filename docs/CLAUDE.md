# 项目文档

## 目录结构

```
C:\codes\python\items_1\
├── .claude/              # Claude Code 配置
│   └── settings.local.json
├── ai_test_1/            # AI 测试项目
├── ai-test_1/            # AI 测试项目
├── CSTVR/                # 视频恢复项目
├── CSTVR_ori/            # 原始 CSTVR
├── lop_repro/            # LOP 复现
├── practice/             # 练习代码
│   ├── decorator_practice/
│   ├── io_practice/
│   └── image_practice/
├── read_paper/           # 论文阅读
│   ├── 1/
│   ├── 2/
│   └── 3/
├── VLMCompression/       # VLM 压缩研究（主要工作）
├── 人工智能课程/          # 课程资料
├── 论文下载整理/          # 论文整理
├── minimax_demo.py       # MiniMax API 演示脚本
├── test_image.png        # 测试图片
└── generated_music.mp3    # 生成的音乐
```

## 主要项目

### VLMCompression
VLM 压缩研究项目（LLaVA/Bunny/InternVL pruning）

### CSTVR
视频恢复相关

## 常用命令

```bash
# Python 环境
conda activate d2l_env
python minimax_demo.py

# Git
git status
git add .
git commit -m "message"
git push
```

## 开发规范

- 写完代码后必须运行验证
- 使用 `PYTHONIOENCODING=utf-8` 解决 Windows 中文输出问题
- minimax_demo.py 包含 MiniMax API 调用示例

## 环境配置

- Python 环境: d2l_env (conda)
- API Key: sk-cp-ALlW5lD3...（Coding Plan）
- API Host: https://api.minimaxi.com

## MiniMax API 验证过的端点

| 功能 | 端点 | 状态 |
|------|------|------|
| 搜索 | POST /v1/coding_plan/search | ✅ |
| VLM | POST /v1/coding_plan/vlm | ✅ |
| 音乐 | POST /v1/music_generation | ✅ |