import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def _http_get_bytes(url: str, timeout_s: int) -> bytes:
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Connection": "close"}, method="GET")
            with urlopen(req, timeout=timeout_s) as resp:
                chunks = []
                while True:
                    buf = resp.read(1024 * 64)
                    if not buf:
                        break
                    chunks.append(buf)
                return b"".join(chunks)
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            if attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            raise
    raise last_err or RuntimeError("下载失败")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_bgm_plan(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "llm", "bgm.plan.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到 bgm.plan.json：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict):
        raise RuntimeError("bgm.plan.json 格式不正确：期望 JSON object")
    return obj


def download_bgm(project_dir: str, timeout_s: int, force: bool) -> str:
    plan = _load_bgm_plan(project_dir)
    selected = plan.get("selected") or {}
    if not isinstance(selected, dict):
        raise RuntimeError("bgm.plan.json 缺少 selected 对象")

    title = str(selected.get("title") or "").strip()
    artist = str(selected.get("artist") or "").strip()
    download_url = str(selected.get("download_url") or "").strip()
    if not download_url:
        raise RuntimeError("bgm.plan.json 的 selected.download_url 为空")

    assets_dir = os.path.join(project_dir, "assets", "bgm")
    _ensure_dir(assets_dir)

    out_audio = os.path.join(assets_dir, "bgm.mp3")
    out_manifest = os.path.join(assets_dir, "manifest.json")

    if not force and os.path.exists(out_audio) and os.path.exists(out_manifest):
        return out_manifest

    data = _http_get_bytes(download_url, timeout_s=timeout_s)
    with open(out_audio, "wb") as fp:
        fp.write(data)

    sha256 = _sha256_bytes(data)
    risks = plan.get("risks") or {}
    license_text = ""
    attribution_text = ""
    if isinstance(risks, dict):
        license_text = str(risks.get("copyright") or "").strip()
        attribution_text = str(risks.get("attribution") or "").strip()

    manifest = {
        "post_id": plan.get("post_id"),
        "post_url": plan.get("post_url"),
        "title": title,
        "artist": artist,
        "download_url": download_url,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "file": {
            "path": os.path.relpath(out_audio, project_dir),
            "bytes": len(data),
            "sha256": sha256,
            "format": "mp3",
        },
        "license": license_text,
        "attribution": attribution_text,
    }
    _write_json(out_manifest, manifest)
    return out_manifest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = download_bgm(project_dir=args.project_dir, timeout_s=int(args.timeout), force=bool(args.force))
    print(json.dumps({"ok": True, "manifest": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

