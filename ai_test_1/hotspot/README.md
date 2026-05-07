# 热点视界 (HotSpot) - 实时热搜聚合平台

## 功能特性

- **9大平台热搜** - 微博、知乎、抖音、百度、TikTok、小红书、抖音热榜、腾讯新闻、快手
- **实时数据** - 每5分钟自动刷新，支持手动刷新
- **话题搜索** - 全平台实时搜索
- **收藏功能** - 收藏感兴趣的话题（持久化存储）
- **评论互动** - 实时评论互动
- **趋势分析** - 话题热度历史追踪
- **数据统计** - 全网话题统计面板

## 项目结构

```
hotspot/
├── backend/
│   ├── main.py          # FastAPI 主服务器 (端口 8888)
│   ├── models.py        # Pydantic 数据模型
│   ├── crawlers.py      # 热搜爬虫 (含模拟数据)
│   ├── database.py      # SQLite 数据库
│   └── requirements.txt # Python 依赖
├── frontend/
│   └── index.html       # 单页应用 (端口 3000)
└── README.md
```

## 快速启动

### 1. 安装依赖

```bash
cd hotspot/backend
pip install -r requirements.txt
```

### 2. 启动后端服务器 (端口 8888)

```bash
cd hotspot/backend
python main.py
```

或使用 uvicorn:

```bash
cd hotspot/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8888
```

### 3. 启动前端服务器 (端口 3000)

```bash
cd hotspot/frontend
python -m http.server 3000
```

## 访问地址

- **前端页面**: http://localhost:3000
- **后端API**: http://localhost:8888
- **API文档**: http://localhost:8888/docs

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/platforms` | GET | 获取所有平台列表 |
| `/api/hotsearch` | GET | 获取所有平台热搜汇总 |
| `/api/hotsearch/{platform}` | GET | 获取指定平台热搜 |
| `/api/refresh` | POST | 手动刷新热搜 |
| `/api/bookmarks` | GET | 获取收藏列表 |
| `/api/bookmarks` | POST | 添加收藏 |
| `/api/bookmarks` | DELETE | 删除收藏 |
| `/api/comments` | GET | 获取评论列表 |
| `/api/comments` | POST | 添加评论 |
| `/api/search` | GET | 搜索热搜话题 |
| `/api/stats` | GET | 获取统计数据 |

## 技术栈

- **后端**: FastAPI + uvicorn + SQLite
- **前端**: 纯 HTML/CSS/JavaScript (无框架)
- **爬虫**: asyncio + httpx (模拟数据)
- **定时任务**: APScheduler

## 注意事项

1. 前端页面会自动检测后端服务端口（8888 或 8000）
2. 热搜数据每5分钟自动刷新
3. 收藏和评论数据保存在本地 SQLite 数据库
4. 如果端口被占用，修改 `main.py` 中的端口号

## 演示数据

系统使用模拟热搜数据，包含以下类型话题：
- 社会热点（#梅西抵达北京#、#多地高温预警#）
- 科技数码（#AI绘画引发争议#、#特斯拉全自动驾驶#）
- 娱乐八卦（#明星塌房现场#、#演员片酬上限#）
- 时事新闻（#神舟十六号发射成功#、#延迟退休最新消息#）
