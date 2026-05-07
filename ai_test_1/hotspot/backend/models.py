"""
热点视界 (HotSpot) - 数据模型
热搜数据结构定义
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Platform(str, Enum):
    """平台枚举"""
    BAIDU = "baidu"
    DOUYIN = "douyin"


# ============================================================
# 平台配置
# ============================================================

PLATFORMS = {
    Platform.BAIDU: {
        "id": "baidu",
        "name": "Baidu",
        "name_cn": "百度",
        "icon": "🔍",
        "color": "#2932E1",
        "url": "https://top.baidu.com",
    },
    Platform.DOUYIN: {
        "id": "douyin",
        "name": "Douyin",
        "name_cn": "抖音",
        "icon": "🎵",
        "color": "#161823",
        "url": "https://www.douyin.com",
    },
}


# ============================================================
# 数据模型
# ============================================================

class HotSearchItem(BaseModel):
    """热搜条目"""
    rank: int
    title: str
    hot_value: Optional[str] = None
    url: str
    is_new: bool = False
    is_hot: bool = False


class PlatformData(BaseModel):
    """单个平台热搜数据"""
    platform: str
    platform_name: str
    platform_icon: str
    platform_color: str
    items: List[HotSearchItem]
    total: int
    update_time: str
    trending_count: int = 0


class HotSearchResponse(BaseModel):
    """热搜响应"""
    platforms: List[PlatformData]
    total_items: int
    update_time: str
    top_topics: List[dict]


class TrendData(BaseModel):
    """话题趋势数据"""
    topic: str
    platform: str
    timestamps: List[str]
    values: List[int]
    change: str  # "up", "down", "stable"
