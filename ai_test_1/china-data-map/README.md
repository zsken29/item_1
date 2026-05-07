# ChinaCity-DataMap 中国城市人均数据地图

交互式可视化中国各城市人均数据（GDP、收入、房价、人口等）的 Web 应用。

## 项目结构

```
china-data-map/
├── backend/               # Python FastAPI 后端
│   ├── app/
│   │   ├── main.py       # API 入口
│   │   ├── router/       # 路由
│   │   ├── models/       # 数据库模型
│   │   └── schemas/      # Pydantic 模型
│   ├── scripts/
│   │   └── init_data.py  # 数据初始化
│   └── requirements.txt
├── frontend/              # Vue3 前端
│   ├── src/
│   │   ├── components/    # 图表组件
│   │   ├── api/          # API 调用
│   │   └── App.vue       # 主组件
│   └── package.json
└── docs/
    └── SPEC.md           # 详细规格文档
```

## 快速启动

### 1. 初始化后端

```bash
cd china-data-map/backend
pip install -r requirements.txt
python scripts/init_data.py  # 初始化数据库
python run.py                # 启动后端 (localhost:8000)
```

### 2. 启动前端

```bash
cd china-data-map/frontend
npm install
npm run dev                   # 启动前端 (localhost:5173)
```

### 3. 访问

打开浏览器访问 http://localhost:5173

## 功能特性

- **中国地图可视化** - 散点图展示各城市数据，悬停查看详情
- **多指标切换** - 人均GDP、人均收入、房价、人口、GDP总量
- **省份/城市排名** - 柱状图展示TOP10排名
- **历年趋势** - 点击城市查看历年数据变化
- **散点图分析** - 展示两个指标的关联关系

## 数据说明

数据来源为国家统计局公开数据整理，部分为基于公开报告的合理估算值。
