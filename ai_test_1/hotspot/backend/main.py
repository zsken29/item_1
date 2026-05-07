"""
热点视界 (HotSpot) - FastAPI 主服务器
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from typing import Optional, Dict, List
import asyncio
import os

from models import Platform, PLATFORMS, PlatformData, HotSearchResponse, HotSearchItem
from database import (
    save_hotsearch, get_bookmarks, add_bookmark, remove_bookmark,
    add_view, get_recent_views, add_comment, get_comments,
    add_search, get_search_history, clear_search_history,
    get_platform_stats, get_top_topics, is_bookmarked, init_database
)
from crawlers import fetch_all, CRAWLER_REGISTRY


# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(title="热点视界 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 热搜缓存
_cache: Dict[str, List[Dict]] = {}
_last_update: datetime = datetime.now()
_cache_lock = asyncio.Lock()


# ============================================================
# 生命周期
# ============================================================

@app.on_event("startup")
async def startup():
    print("热点视界 API 启动中...")
    init_database()
    await refresh()
    print("启动完成 ✅")


# ============================================================
# 核心逻辑
# ============================================================

async def refresh():
    """刷新所有平台热搜"""
    global _cache, _last_update
    async with _cache_lock:
        _cache = await fetch_all()
        _last_update = datetime.now()
        for platform, items in _cache.items():
            if items:
                save_hotsearch(platform, items)


def _build_response(platform: Optional[str] = None) -> dict:
    """构建热搜响应"""
    global _cache, _last_update

    if platform:
        items = _cache.get(platform, [])
        cfg = PLATFORMS.get(Platform(platform))
        if not cfg:
            raise HTTPException(status_code=404, detail="平台不存在")
        return {
            "platform": platform,
            "platform_name": cfg["name_cn"],
            "platform_icon": cfg["icon"],
            "platform_color": cfg["color"],
            "items": items,
            "total": len(items),
            "update_time": _last_update.isoformat(),
            "trending_count": sum(1 for i in items if i.get("is_hot")),
        }

    # 所有平台
    all_items = []
    platforms_data = []

    for p, items in _cache.items():
        all_items.extend(items)
        cfg = PLATFORMS.get(Platform(p))
        if cfg:
            platforms_data.append({
                "platform": p,
                "platform_name": cfg["name_cn"],
                "platform_icon": cfg["icon"],
                "platform_color": cfg["color"],
                "items": items,
                "total": len(items),
                "update_time": _last_update.isoformat(),
                "trending_count": sum(1 for i in items if i.get("is_hot")),
            })

    # 全网最热（按 is_hot 标记和排名）
    top = sorted(all_items, key=lambda x: (x.get("is_hot", False), -x.get("rank", 999)), reverse=True)[:10]

    return {
        "platforms": platforms_data,
        "total_items": len(all_items),
        "update_time": _last_update.isoformat(),
        "top_topics": [{"title": t["title"], "platform": t.get("platform", ""), "rank": t["rank"]} for t in top],
    }


# ============================================================
# API 路由
# ============================================================

@app.get("/")
def root():
    return {"name": "热点视界 API", "version": "1.0.0"}


@app.get("/api/platforms")
def list_platforms():
    """平台列表"""
    return [
        {
            "id": p.value,
            "name": cfg["name"],
            "name_cn": cfg["name_cn"],
            "icon": cfg["icon"],
            "color": cfg["color"],
            "url": cfg["url"],
            "item_count": len(_cache.get(p.value, [])),
        }
        for p, cfg in PLATFORMS.items()
    ]


@app.get("/api/hotsearch")
def hotsearch():
    """所有平台热搜汇总"""
    return _build_response()


@app.get("/api/hotsearch/{platform}")
def platform_hotsearch(platform: str):
    """指定平台热搜"""
    return _build_response(platform)


@app.post("/api/refresh")
def refresh_api(background_tasks: BackgroundTasks):
    """手动刷新"""
    background_tasks.add_task(refresh)
    return {"message": "刷新已启动", "timestamp": datetime.now().isoformat()}


# ============================================================
# 收藏
# ============================================================

@app.get("/api/bookmarks")
def bookmarks():
    return {"bookmarks": get_bookmarks(), "total": len(get_bookmarks())}


@app.post("/api/bookmarks")
def bookmark_create(item: dict):
    ok = add_bookmark(item.get("topic", ""), item.get("platform", ""), item.get("url", ""))
    if ok:
        return {"message": "收藏成功", "bookmarked": True}
    raise HTTPException(status_code=400, detail="收藏失败")


@app.delete("/api/bookmarks")
def bookmark_delete(topic: str = Query(...), platform: str = Query(...)):
    ok = remove_bookmark(topic, platform)
    return {"message": "已取消" if ok else "失败", "bookmarked": not ok}


@app.get("/api/bookmarks/check")
def bookmark_check(topic: str = Query(...), platform: str = Query(...)):
    return {"bookmarked": is_bookmarked(topic, platform)}


# ============================================================
# 浏览记录
# ============================================================

@app.post("/api/views")
def view_create(item: dict):
    add_view(item.get("topic", ""), item.get("platform", ""))
    return {"message": "记录成功"}


@app.get("/api/views")
def views(limit: int = Query(20, ge=1, le=100)):
    return {"views": get_recent_views(limit)}


# ============================================================
# 评论
# ============================================================

@app.get("/api/comments")
def comments(topic: Optional[str] = None, platform: Optional[str] = None,
            limit: int = Query(50, ge=1, le=200)):
    cs = get_comments(topic=topic, platform=platform, limit=limit)
    return {"comments": cs, "total": len(cs)}


@app.post("/api/comments")
def comment_create(item: dict):
    text = item.get("comment", "")
    if not text:
        raise HTTPException(status_code=400, detail="评论内容不能为空")
    ok = add_comment(item.get("topic", "全网话题"), item.get("platform", "baidu"), text)
    if ok:
        return {"message": "评论成功"}
    raise HTTPException(status_code=400, detail="评论失败")


# ============================================================
# 搜索
# ============================================================

@app.get("/api/search/history")
def search_history(limit: int = Query(10, ge=1, le=50)):
    return {"history": get_search_history(limit)}


@app.post("/api/search/history")
def search_history_create(keyword: str = Query(...)):
    add_search(keyword)
    return {"message": "记录成功"}


@app.delete("/api/search/history")
def search_history_clear():
    clear_search_history()
    return {"message": "已清空"}


@app.get("/api/search")
def search(q: str = Query(...), platform: Optional[str] = None,
           limit: int = Query(20, ge=1, le=100)):
    add_search(q)
    results = []
    for p, items in _cache.items():
        if platform and p != platform:
            continue
        for item in items:
            if q.lower() in item.get("title", "").lower():
                cfg = PLATFORMS.get(Platform(p), {})
                item = dict(item)
                item["platform_name"] = cfg.get("name_cn", p)
                item["platform_icon"] = cfg.get("icon", "")
                results.append(item)
    results.sort(key=lambda x: (x.get("is_hot", False), -x.get("rank", 999)), reverse=True)
    return {"query": q, "results": results[:limit], "total": len(results)}


# ============================================================
# 趋势 & 统计
# ============================================================

@app.get("/api/trend/{topic}")
def trend(topic: str, hours: int = Query(24, ge=1, le=168)):
    data = get_top_topics(hours=hours, limit=10)
    return {"topic": topic, "data": data}


@app.get("/api/stats")
def stats():
    return {
        "platform_stats": get_platform_stats(24),
        "top_topics": get_top_topics(24, 20),
        "cache": {
            "last_update": _last_update.isoformat(),
            "total_platforms": len(_cache),
            "total_items": sum(len(v) for v in _cache.values()),
        },
    }


@app.get("/api/history")
def history(platform: Optional[str] = None, hours: int = Query(24, ge=1, le=168)):
    from database import get_history
    return {"history": get_history(platform, hours)}


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================
# 静态文件
# ============================================================

static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False)
