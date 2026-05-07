"""
热点视界 (HotSpot) - 数据库管理
使用SQLite存储热搜历史和用户交互
"""
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional
from contextlib import contextmanager
import json

DATABASE_PATH = "hotspot.db"


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """数据库上下文管理器"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库表"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 热搜历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hotsearch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                rank INTEGER NOT NULL,
                title TEXT NOT NULL,
                hot_value TEXT,
                url TEXT,
                is_new INTEGER DEFAULT 0,
                is_hot INTEGER DEFAULT 0,
                is_focus INTEGER DEFAULT 0,
                is_recommend INTEGER DEFAULT 0,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform, title, timestamp)
            )
        """)

        # 趋势数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trend_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                platform TEXT NOT NULL,
                rank INTEGER NOT NULL,
                hot_value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic, platform, timestamp)
            )
        """)

        # 用户收藏表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                platform TEXT NOT NULL,
                url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic, platform)
            )
        """)

        # 用户浏览历史
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                platform TEXT NOT NULL,
                viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 用户评论表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                platform TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 搜索历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_platform ON hotsearch_history(platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON hotsearch_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trend_topic ON trend_data(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trend_timestamp ON trend_data(timestamp)")

        conn.commit()
        print("✅ 数据库初始化完成")


# ============ 热搜历史操作 ============

def save_hotsearch(platform: str, items: List[dict]):
    """保存热搜数据"""
    with get_db() as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for item in items:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO hotsearch_history
                    (platform, rank, title, hot_value, url, is_new, is_hot, is_focus, is_recommend, category, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    platform,
                    str(item.get('rank', 0)),  # 转为字符串避免类型错误
                    str(item.get('title', '')),
                    str(item.get('hot_value', '')),
                    str(item.get('url', '')),
                    int(item.get('is_new', False)),
                    int(item.get('is_hot', False)),
                    int(item.get('is_focus', False)),
                    int(item.get('is_recommend', False)),
                    str(item.get('category', '')),
                    timestamp
                ))
            except Exception as e:
                print(f"保存热搜失败: {e}")

        conn.commit()


def get_history(platform: Optional[str] = None, hours: int = 24) -> List[dict]:
    """获取历史热搜"""
    with get_db() as conn:
        cursor = conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        if platform:
            cursor.execute("""
                SELECT * FROM hotsearch_history
                WHERE platform = ? AND timestamp > ?
                ORDER BY timestamp DESC, rank ASC
            """, (platform, since))
        else:
            cursor.execute("""
                SELECT * FROM hotsearch_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC, rank ASC
            """, (since,))

        return [dict(row) for row in cursor.fetchall()]


def get_topic_trend(topic: str, hours: int = 24) -> List[dict]:
    """获取话题趋势"""
    with get_db() as conn:
        cursor = conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            SELECT timestamp, rank, hot_value
            FROM hotsearch_history
            WHERE title LIKE ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (f"%{topic}%", since))

        return [dict(row) for row in cursor.fetchall()]


# ============ 收藏操作 ============

def add_bookmark(topic: str, platform: str, url: str = "") -> bool:
    """添加收藏"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_bookmarks (topic, platform, url, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (topic, platform, url))
            conn.commit()
        return True
    except Exception as e:
        print(f"添加收藏失败: {e}")
        return False


def remove_bookmark(topic: str, platform: str) -> bool:
    """移除收藏"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM user_bookmarks WHERE topic = ? AND platform = ?
            """, (topic, platform))
            conn.commit()
        return True
    except Exception as e:
        print(f"移除收藏失败: {e}")
        return False


def get_bookmarks() -> List[dict]:
    """获取所有收藏"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM user_bookmarks ORDER BY created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def is_bookmarked(topic: str, platform: str) -> bool:
    """检查是否已收藏"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM user_bookmarks WHERE topic = ? AND platform = ?
        """, (topic, platform))
        return cursor.fetchone() is not None


# ============ 浏览历史操作 ============

def add_view(topic: str, platform: str):
    """添加浏览记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_views (topic, platform) VALUES (?, ?)
        """, (topic, platform))
        conn.commit()


def get_recent_views(limit: int = 20) -> List[dict]:
    """获取最近浏览"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT topic, platform, MAX(viewed_at) as viewed_at
            FROM user_views
            GROUP BY topic, platform
            ORDER BY viewed_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


# ============ 评论操作 ============

def add_comment(topic: str, platform: str, comment: str) -> bool:
    """添加评论"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_comments (topic, platform, comment) VALUES (?, ?, ?)
            """, (topic, platform, comment))
            conn.commit()
        return True
    except Exception as e:
        print(f"添加评论失败: {e}")
        return False


def get_comments(topic: str = None, platform: str = None, limit: int = 50) -> List[dict]:
    """获取评论"""
    with get_db() as conn:
        cursor = conn.cursor()

        if topic and platform:
            cursor.execute("""
                SELECT * FROM user_comments
                WHERE topic = ? AND platform = ?
                ORDER BY created_at DESC LIMIT ?
            """, (topic, platform, limit))
        elif topic:
            cursor.execute("""
                SELECT * FROM user_comments
                WHERE topic = ?
                ORDER BY created_at DESC LIMIT ?
            """, (topic, limit))
        else:
            cursor.execute(f"""
                SELECT * FROM user_comments
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))

        return [dict(row) for row in cursor.fetchall()]


# ============ 搜索历史操作 ============

def add_search(keyword: str):
    """添加搜索记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (keyword) VALUES (?)
        """, (keyword,))
        conn.commit()


def get_search_history(limit: int = 10) -> List[dict]:
    """获取搜索历史"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT keyword, MAX(created_at) as created_at
            FROM search_history
            GROUP BY keyword
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def clear_search_history():
    """清空搜索历史"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history")
        conn.commit()


# ============ 统计分析 ============

def get_platform_stats(hours: int = 24) -> dict:
    """获取平台统计"""
    with get_db() as conn:
        cursor = conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            SELECT platform, COUNT(*) as count, MAX(timestamp) as last_update
            FROM hotsearch_history
            WHERE timestamp > ?
            GROUP BY platform
        """, (since,))

        return {row['platform']: {
            'count': row['count'],
            'last_update': row['last_update']
        } for row in cursor.fetchall()}


def get_top_topics(hours: int = 24, limit: int = 10) -> List[dict]:
    """获取最热话题"""
    with get_db() as conn:
        cursor = conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            SELECT title, platform, COUNT(*) as frequency,
                   AVG(rank) as avg_rank,
                   MIN(rank) as best_rank
            FROM hotsearch_history
            WHERE timestamp > ?
            GROUP BY title
            ORDER BY frequency DESC, avg_rank ASC
            LIMIT ?
        """, (since, limit))

        return [dict(row) for row in cursor.fetchall()]


# 初始化数据库
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

init_database()
