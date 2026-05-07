"""
MiniMax Coding Plan API Demo Script

Verified working endpoints:
  - Search:      POST /v1/coding_plan/search
  - VLM:         POST /v1/coding_plan/vlm
  - Music Gen:   POST /v1/music_generation  (requires 'lyrics' param!)

Unverified (need more params or different endpoint):
  - Lyrics Gen:  POST /v1/lyrics_generation (status 400 - invalid params)
  - Cover Gen:   /v1/music_generation/cover (status 404)

Test image: test_image.png (downloaded from PyTorch Hub)
"""

import base64
import json
import os
import requests

# ========== 配置 ==========
API_KEY = "sk-cp-ALlW5lD3StgyNRUCc4QnaUNJs8WngxIiEswevo9TcclPrwapHd5QwSLL1VXMuBZoXqg6T4TpiwKrttEb27ko0Qn4NmaGflgvjvR6hX05GdqsLfQI2ZXlvh8"
API_HOST = "https://api.minimaxi.com"


# ========== 1. 网络搜索 (coding-plan-search) ==========
def web_search(query: str, num_results: int = 5):
    """MiniMax Search API - returns organic search results"""
    url = f"{API_HOST}/v1/coding_plan/search"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"q": query, "num_results": num_results}

    print(f"\n=== [Search] ===")
    print(f"Query: {query}")
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        results = data.get("organic", [])
        print(f"Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.get('title', 'N/A')}")
            print(f"     URL: {r.get('link', 'N/A')}")
    else:
        print(f"Error {response.status_code}: {response.text}")

    return response


# ========== 2. 图像理解 (coding-plan-vlm) ==========
def understand_image(image_path: str, prompt: str = "描述这张图片"):
    """MiniMax VLM API - analyze images with AI"""
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None

    url = f"{API_HOST}/v1/coding_plan/vlm"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    # Detect image type
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    payload = {"prompt": prompt, "image_url": f"data:{mime_type};base64,{img_data}"}

    print(f"\n=== [VLM Image Analysis] ===")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        data = response.json()
        content = data.get("content", "No content returned")
        print(f"Result: {content}")
    else:
        print(f"Error {response.status_code}: {response.text}")

    return response


# ========== 3. 音乐生成 (music-2.6) ==========
def generate_music(prompt: str, lyrics: str, duration: int = 30):
    """MiniMax Music Generation API

    Args:
        prompt: Music description/style
        lyrics: Song lyrics (REQUIRED! API will error without it)
        duration: Duration in seconds
    """
    url = f"{API_HOST}/v1/music_generation"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "music-2.6",
        "prompt": prompt,
        "lyrics": lyrics,  # Required!
        "duration": duration,
    }

    print(f"\n=== [Music Generation] ===")
    print(f"Prompt: {prompt}")
    print(f"Lyrics: {lyrics[:50]}...")

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        data = response.json()
        audio_hex = data.get("data", {}).get("audio", "")
        if audio_hex:
            # Audio is hex-encoded, not base64
            audio_bytes = bytes.fromhex(audio_hex)
            output_path = "generated_music.mp3"
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"SUCCESS: Audio saved to {output_path} ({len(audio_bytes):,} bytes)")
            status = data.get("data", {}).get("status", "")
            print(f"Status: {status}")
        else:
            print(f"Response: {json.dumps(data, ensure_ascii=False)[:500]}")
    else:
        print(f"Error {response.status_code}: {response.text}")

    return response


# ========== 4. 查询用量 ==========
def check_usage():
    """Query Coding Plan usage - endpoint needs investigation"""
    # Common endpoints that might work
    endpoints = [
        "/v1/cp/usage",
        "/api/v1/cp/usage",
        "/v1/usage",
        "/api/usage",
    ]

    print(f"\n=== [Usage Check] ===")
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for ep in endpoints:
        r = requests.get(f"{API_HOST}{ep}", headers=headers, timeout=10)
        print(f"GET {ep}: {r.status_code}")
        if r.status_code == 200:
            print(f"Response: {r.text[:200]}")

    return None


# ========== 主函数 ==========
if __name__ == "__main__":
    # 1. 搜索测试
    web_search("Python programming")

    # 2. VLM 图像理解测试
    if os.path.exists("test_image.png"):
        understand_image("test_image.png", "描述图片中的场景和人物")
    else:
        print("\n[VLM] test_image.png not found, skipping")

    # 3. 音乐生成测试
    generate_music(
        prompt="欢快的电子音乐",
        lyrics="太阳当空照，花儿对我笑，小鸟说早早早，我们一起去学校"
    )

    # 4. 用量查询
    check_usage()

    print("\n=== Demo Complete ===")