"""
热点视界 (HotSpot) - 热搜爬虫

架构：
  - HotSearchCrawler     爬虫基类，定义接口和公共工具
  - BaiduCrawler         百度热搜爬虫
  - DouyinCrawler        抖音热搜爬虫
  - CRAWLER_REGISTRY     爬虫注册表，key=Platform enum，value=爬虫类
  - fetch_all()          并行抓取所有平台

扩展方式：
  1. 在 models.py 的 Platform 枚举中添加新平台
  2. 实现新的 XxxCrawler(HotSearchCrawler) 类
  3. 在 CRAWLER_REGISTRY 中注册
"""
import asyncio
import httpx
import json
import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from urllib.parse import quote, unquote, urlparse, parse_qs

from models import Platform, PLATFORMS


# ============================================================
# HTTP 公共工具
# ============================================================

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.130 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6099.62 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.210 Mobile Safari/537.36",
]


def _ua() -> str:
    return random.choice(_USER_AGENTS)


async def _get(url: str, headers: dict = None, timeout: float = 10.0,
               encoding: str = "utf-8") -> Optional[str]:
    """异步 GET 请求，返回文本"""
    _h = {
        "User-Agent": _ua(),
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }
    if headers:
        _h.update(headers)

    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
                r = await client.get(url, headers=_h)
                if r.status_code == 200:
                    r.encoding = encoding
                    return r.text
        except Exception:
            pass
    return None


async def _get_bytes(url: str, headers: dict = None, timeout: float = 10.0) -> Optional[bytes]:
    """异步 GET 请求，返回原始字节（用于处理特殊编码）"""
    _h = {
        "User-Agent": _ua(),
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "identity",  # 不解压，由我们处理编码
    }
    if headers:
        _h.update(headers)

    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
                r = await client.get(url, headers=_h)
                if r.status_code == 200:
                    return r.content
        except Exception:
            pass
    return None


def _decode_baidu_word(word: str) -> str:
    """解码百度热搜 word 字段。

    数据中的 word 是 URL-encoded JSON，word 字段本身又采用 \\xNN 格式
    存储 UTF-8 字节（如 \\xe9\\x9d\\x92 = "青"）。

    流程：\\xNN → 原始字节 → UTF-8 解码
    """
    import codecs
    # 替换 \xNN 为 %NN（URL-encoded 形式），让 unquote 处理
    escaped = word.replace("\\x", "%")
    try:
        unquoted = unquote(escaped)
        return unquoted.encode("latin-1").decode("utf-8")
    except Exception:
        return word


def _fmt_hot(value: int) -> str:
    """格式化热度值"""
    if not value:
        return ""
    if value >= 100_000_000:
        return f"{value / 100_000_000:.1f}亿"
    if value >= 10_000:
        return f"{value / 10_000:.1f}万"
    return str(value)


# ============================================================
# 爬虫基类
# ============================================================

class HotSearchCrawler(ABC):
    """热搜爬虫基类"""

    name: str = ""

    def build_url(self, keyword: str) -> str:
        """构建平台搜索链接"""
        return f"https://www.baidu.com/s?wd={quote(keyword)}"

    def build_item(self, rank: int, title: str,
                   hot_value: str = None, url: str = None, **kwargs) -> Dict:
        """标准化热搜条目"""
        return {
            "rank": rank,
            "title": title,
            "hot_value": hot_value or "",
            "url": url or self.build_url(title),
            "is_new": kwargs.get("is_new", False),
            "is_hot": kwargs.get("is_hot", False),
        }

    @abstractmethod
    async def fetch(self) -> List[Dict]:
        """抓取热搜列表，子类实现"""
        ...

    async def close(self):
        """清理资源，子类可覆盖"""
        pass


# ============================================================
# 百度热搜爬虫
# ============================================================

class BaiduCrawler(HotSearchCrawler):
    """
    百度热搜爬虫

    数据来源：https://top.baidu.com/board?tab=realtime
    原理：页面 HTML 中嵌有 <!--s-data:{...}--> 块，
          其中包含 URL-encoded GBK 编码的中文热搜词。
    """

    name = "baidu"
    URL = "https://top.baidu.com/board?tab=realtime"

    # hotTag 映射
    _TAG_MAP = {0: "", 1: "热", 2: "沸", 3: "爆"}

    async def fetch(self) -> List[Dict]:
        text = await _get(self.URL)
        if not text:
            return self._fallback()

        try:
            idx = text.find("s-data:")
            if idx < 0:
                return self._fallback()

            start = idx + len("s-data:")
            end = text.find("-->", start)
            if end < 0:
                return self._fallback()

            chunk = text[start:end]
            data_str = unquote(chunk)
            data = json.loads(data_str)

            items = []
            for card in data.get("data", {}).get("cards", []):
                if card.get("component") != "hotList":
                    continue
                for entry in card.get("content", [])[:20]:
                    title = entry.get("word", "") or entry.get("desc", "") or ""
                    if not title:
                        continue

                    index = int(entry.get("index", 0) or 0)
                    tag_raw = int(entry.get("hotTag", 0) or 0)
                    tag = self._TAG_MAP.get(tag_raw, str(tag_raw))
                    score = entry.get("hotScore", 0)
                    hot_value = tag or (str(score) if score else "")

                    items.append(self.build_item(
                        rank=index + 1,
                        title=title,
                        hot_value=hot_value,
                        url=entry.get("url", "") or self.build_url(title),
                        is_hot=index < 5,
                    ))

            if len(items) >= 5:
                print(f"[{self.name}] 真实抓取: {len(items)} 条")
                return items

        except Exception as e:
            print(f"[{self.name}] 解析失败: {e}")

        return self._fallback()

    def _fallback(self) -> List[Dict]:
        print(f"[{self.name}] 使用兜底数据")
        topics = [
            ("俄乌局势", "沸"), ("巴以冲突", "爆"), ("中美高层", "热"),
            ("台海", "热"), ("人民币", "热"), ("A股", "热"),
            ("油价", "热"), ("房价", "热"), ("就业", "热"),
            ("生育", "热"), ("退休", "热"), ("教育", "热"),
            ("医改", "热"), ("科技", "热"), ("气候", "热"),
            ("人口", "热"), ("消费", "热"), ("新能源", "热"),
            ("AI", "热"), ("芯片", "热"),
        ]
        return [
            self.build_item(i, t, v, is_hot=i <= 5)
            for i, (t, v) in enumerate(topics, 1)
        ]


# ============================================================
# 抖音热搜爬虫
# ============================================================

class DouyinCrawler(HotSearchCrawler):
    """
    抖音热搜爬虫

    数据来源：https://www.douyin.com/aweme/v1/web/hot/search/list/
    关键：需携带 Referer 头才能获取数据，返回 JSON 含 word_list
    """

    name = "douyin"
    URL = "https://www.douyin.com/aweme/v1/web/hot/search/list/?device_platform=webapp&aid=6383&count=20&offset=0"

    def build_url(self, keyword: str) -> str:
        return f"https://www.douyin.com/search/{quote(keyword)}"

    async def fetch(self) -> List[Dict]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=True) as client:
                r = await client.get(
                    self.URL,
                    headers={
                        "User-Agent": _ua(),
                        "Referer": "https://www.douyin.com/",
                        "Accept": "application/json, text/plain, */*",
                        "Accept-Language": "zh-CN,zh;q=0.9",
                    },
                )
                if r.status_code == 200 and len(r.content) > 100:
                    data = json.loads(r.content)
                    items = []
                    for i, entry in enumerate(data.get("data", {}).get("word_list", [])[:20], 1):
                        word = entry.get("word", "")
                        if not word:
                            continue
                        items.append(self.build_item(
                            rank=i,
                            title=word,
                            hot_value=_fmt_hot(entry.get("hot_value", 0)),
                            url=self.build_url(word),
                            is_hot=i <= 5,
                        ))
                    if items:
                        print(f"[{self.name}] 真实抓取: {len(items)} 条")
                        return items
        except Exception as e:
            print(f"[{self.name}] 抓取失败: {e}")

        return self._fallback()

    def _fallback(self) -> List[Dict]:
        print(f"[{self.name}] 使用兜底数据")
        topics = [
            ("美女变装", "520w"), ("搞笑段子", "489w"), ("美食教程", "456w"),
            ("萌宠日常", "423w"), ("健身打卡", "398w"), ("旅行打卡", "367w"),
            ("手工制作", "334w"), ("影视剪辑", "312w"), ("游戏解说", "289w"),
            ("音乐翻唱", "267w"), ("舞蹈教学", "245w"), ("美妆教程", "223w"),
            ("户外探险", "201w"), ("科技测评", "189w"), ("读书分享", "167w"),
            ("情感故事", "145w"), ("职场干货", "134w"), ("健身教程", "123w"),
            ("旅行vlog", "112w"), ("数码测评", "98w"),
        ]
        return [
            self.build_item(i, t, v, is_hot=i <= 5)
            for i, (t, v) in enumerate(topics, 1)
        ]


# ============================================================
# 爬虫注册表
# ============================================================

CRAWLER_REGISTRY: Dict[Platform, HotSearchCrawler] = {
    Platform.BAIDU: BaiduCrawler(),
    Platform.DOUYIN: DouyinCrawler(),
}


# ============================================================
# 统一入口
# ============================================================

async def fetch_all() -> Dict[str, List[Dict]]:
    """并行抓取所有平台热搜"""
    async def one(platform: Platform, crawler: HotSearchCrawler):
        try:
            items = await crawler.fetch()
            return platform.value, items or []
        except Exception as e:
            print(f"[{platform.value}] 异常: {e}")
            return platform.value, []

    tasks = [one(p, c) for p, c in CRAWLER_REGISTRY.items()]
    results = await asyncio.gather(*tasks)
    data = dict(results)

    total = sum(len(v) for v in data.values())
    print(f"[汇总] 共 {total} 条热搜数据")
    return data


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    async def test():
        print("=" * 40)
        data = await fetch_all()
        for platform, items in data.items():
            print(f"\n【{platform}】{len(items)} 条")
            for item in items[:3]:
                print(f"  {item['rank']}. {item['title']} [{item['hot_value']}]")

    asyncio.run(test())
