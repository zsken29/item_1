# ChinaCity-DataMap 项目规格说明

## 1. 项目概述

中国城市人均数据地图 - 展示中国各地级市的人均GDP、人均收入、房价、人口等数据的交互式可视化平台。

## 2. 技术架构

| 层级 | 技术 |
|------|------|
| 前端 | Vue3 + ECharts |
| 后端 | Python FastAPI |
| 数据库 | SQLite |

## 3. 数据指标

- 人均GDP（万元）
- 人均可支配收入（元）
- 商品房均价（元/㎡）
- 常住人口（万人）
- GDP总量（亿元）

## 4. API 接口

- `GET /api/cities` - 城市列表
- `GET /api/cities/with-stats` - 带数据的城市列表
- `GET /api/ranking/province` - 省份排名
- `GET /api/ranking/city` - 城市排名
- `GET /api/trend/{city_id}` - 城市历年趋势
- `GET /api/scatter` - 散点图数据

## 5. 页面功能

- 顶部：标题 + 指标选择 + 年份选择
- 主体：中国地图（散点热力图）
- 下部：4个图表（省份排名、城市排名、历年趋势、散点图）
- 右侧浮层：选中城市数据详情
