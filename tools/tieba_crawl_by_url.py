import argparse
import csv
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from http.cookiejar import CookieJar
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class CrawlError(RuntimeError):
    pass


@dataclass
class Floor:
    index: int
    author: str
    time: str
    text: str
    images: List[str]


def _extract_post_id(post_url: str) -> str:
    m = re.search(r"/p/(\d+)", post_url)
    if not m:
        raise CrawlError(f"无法从 URL 提取 post_id：{post_url}")
    return m.group(1)


def _normalize_post_url(post_url: str, lz_only: bool) -> str:
    u = urlparse(post_url)
    q = parse_qs(u.query, keep_blank_values=True)
    if lz_only:
        q["see_lz"] = ["1"]
    q.pop("pn", None)
    query = urlencode({k: v[-1] if v else "" for k, v in q.items()})
    return urlunparse((u.scheme or "https", u.netloc or "tieba.baidu.com", u.path, "", query, ""))


def _page_url(base_post_url: str, pn: int) -> str:
    u = urlparse(base_post_url)
    q = parse_qs(u.query, keep_blank_values=True)
    q["pn"] = [str(pn)]
    query = urlencode({k: v[-1] if v else "" for k, v in q.items()})
    return urlunparse((u.scheme, u.netloc, u.path, "", query, ""))

def _build_opener() -> Tuple[CookieJar, object]:
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    return jar, opener


def _http_get(opener, url: str, user_agent: str, referer: str, timeout_s: int = 25) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": referer,
            "Connection": "close",
        },
        method="GET",
    )
    with opener.open(req, timeout=timeout_s) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")


def _detect_block(html_text: str) -> Optional[str]:
    t = html_text
    signals = [
        ("secure-input", "检测到安全验证页面（secure-input），可能被风控拦截"),
        ("百度安全验证", "检测到“百度安全验证”页面，可能需要降低频率或更换网络"),
        ("访问频繁", "检测到“访问频繁”提示，建议降低抓取频率并重试"),
        ("请输入验证码", "检测到验证码提示，建议登录后抓取或降低频率"),
    ]
    for key, msg in signals:
        if key in t:
            return msg
    return None


def _extract_title(html_text: str) -> str:
    m = re.search(r"<title>\s*(.*?)\s*</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = html.unescape(m.group(1))
        title = re.sub(r"\s+", " ", title).strip()
        title = re.sub(r"_百度贴吧.*$", "", title).strip()
        return title
    return ""


def _strip_tags(s: str) -> str:
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n", s)
    s = re.sub(r"(?is)<p[^>]*>", "", s)
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", "", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", "", s)
    s = re.sub(r"(?is)<[^>]+>", "", s)
    s = html.unescape(s)
    s = s.replace("\u200b", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _parse_l_post_blocks(page_html: str) -> List[str]:
    blocks = re.findall(r'(?is)<div[^>]+class="[^"]*\bl_post\b[^"]*"[^>]*>.*?</div>\s*</div>\s*</div>', page_html)
    if blocks:
        return blocks
    blocks = re.findall(r"(?is)<div[^>]+class='[^']*\bl_post\b[^']*'[^>]*>.*?</div>\s*</div>\s*</div>", page_html)
    return blocks


def _parse_data_field(block_html: str) -> Optional[Dict]:
    m = re.search(r"data-field='(.*?)'", block_html, re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r'data-field="(.*?)"', block_html, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    raw = html.unescape(raw)
    raw = raw.replace("&quot;", '"')
    try:
        return json.loads(raw)
    except Exception:
        return None


def _parse_floor_from_block(block_html: str) -> Optional[Floor]:
    data = _parse_data_field(block_html) or {}
    author = (data.get("author") or {}).get("user_name") or ""
    content = data.get("content") or {}
    index = int(content.get("post_no") or 0) if str(content.get("post_no") or "").isdigit() else 0
    time_str = str(content.get("date") or "")

    m = re.search(r'(?is)<div[^>]+class="[^"]*\bd_post_content\b[^"]*"[^>]*>(.*?)</div>', block_html)
    if not m:
        m = re.search(r"(?is)<div[^>]+class='[^']*\bd_post_content\b[^']*'[^>]*>(.*?)</div>", block_html)
    inner = m.group(1) if m else ""
    text = _strip_tags(inner)

    imgs = re.findall(r'(?is)<img[^>]+class="[^"]*\bBDE_Image\b[^"]*"[^>]+src="([^"]+)"', block_html)
    if not imgs:
        imgs = re.findall(r"(?is)<img[^>]+class='[^']*\bBDE_Image\b[^']*'[^>]+src='([^']+)'", block_html)
    imgs = [i.strip() for i in imgs if i and i.strip()]

    if not (text or imgs):
        return None

    return Floor(
        index=index,
        author=author,
        time=time_str,
        text=text,
        images=imgs,
    )


def crawl_post(post_url: str, max_pages: int, lz_only: bool, user_agent: str, sleep_s: float) -> Tuple[str, str, List[Floor]]:
    post_id = _extract_post_id(post_url)
    base_url = _normalize_post_url(post_url, lz_only=lz_only)

    title = ""
    floors: List[Floor] = []
    seen = set()

    _, opener = _build_opener()
    try:
        _http_get(opener, "https://tieba.baidu.com/", user_agent=user_agent, referer="https://tieba.baidu.com/", timeout_s=15)
    except Exception:
        pass

    for pn in range(1, max_pages + 1):
        url = _page_url(base_url, pn)
        last_err: Optional[Exception] = None
        page_html = ""
        for _ in range(2):
            try:
                page_html = _http_get(opener, url, user_agent=user_agent, referer=base_url, timeout_s=25)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6)
        if last_err:
            raise last_err

        block_msg = _detect_block(page_html)
        if block_msg:
            raise CrawlError(f"pn={pn} 抓取被拦截：{block_msg}")

        if not title:
            title = _extract_title(page_html)

        blocks = _parse_l_post_blocks(page_html)
        if not blocks:
            raise CrawlError(f"pn={pn} 未解析到楼层块，可能页面结构变化或被风控拦截")

        for b in blocks:
            floor = _parse_floor_from_block(b)
            if not floor:
                continue
            key = (floor.index, floor.author, floor.time, floor.text[:80])
            if key in seen:
                continue
            seen.add(key)
            floors.append(floor)

        if sleep_s > 0:
            time.sleep(sleep_s)

    floors = [f for f in floors if f.index > 0]
    floors.sort(key=lambda x: x.index)
    return post_id, title, floors


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_outputs(post_url: str, post_id: str, title: str, floors: List[Floor], out_dir: str) -> Tuple[str, str]:
    source_dir = os.path.join(out_dir, post_id, "source")
    _ensure_dir(source_dir)

    raw_path = os.path.join(source_dir, "raw.json")
    csv_path = os.path.join(source_dir, "details.csv")

    raw_obj = {
        "post_url": post_url,
        "post_id": post_id,
        "title": title,
        "floors": [
            {
                "index": f.index,
                "author": f.author,
                "time": f.time,
                "text": f.text,
                "images": f.images,
            }
            for f in floors
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(raw_path, "w", encoding="utf-8") as fp:
        json.dump(raw_obj, fp, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["post_url", "post_id", "title", "author", "section", "index", "like", "comment_author", "text", "images", "time"])
        for f in floors:
            w.writerow(
                [
                    post_url,
                    post_id,
                    title,
                    f.author,
                    "floor",
                    f.index,
                    "",
                    "",
                    f.text,
                    ";".join(f.images),
                    f.time,
                ]
            )

    return raw_path, csv_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--post-url", required=True)
    p.add_argument("--max-pages", type=int, default=3)
    p.add_argument("--scope", choices=["author", "all"], default="author")
    p.add_argument("--user-agent", default=DEFAULT_UA)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "projects"))
    args = p.parse_args(argv)

    post_url = args.post_url.strip()
    max_pages = max(1, int(args.max_pages))
    lz_only = args.scope == "author"

    try:
        post_id, title, floors = crawl_post(
            post_url=post_url,
            max_pages=max_pages,
            lz_only=lz_only,
            user_agent=args.user_agent,
            sleep_s=max(0.0, float(args.sleep)),
        )
        if not floors:
            raise CrawlError("抓取到 0 个楼层，可能被风控拦截或解析失败")
        raw_path, csv_path = write_outputs(
            post_url=post_url,
            post_id=post_id,
            title=title,
            floors=floors,
            out_dir=args.out_dir,
        )
        print(json.dumps({"ok": True, "post_id": post_id, "title": title, "floors": len(floors), "raw": raw_path, "csv": csv_path}, ensure_ascii=False))
        return 0
    except CrawlError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 2
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"未知错误：{e.__class__.__name__}: {e}"}, ensure_ascii=False))
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
