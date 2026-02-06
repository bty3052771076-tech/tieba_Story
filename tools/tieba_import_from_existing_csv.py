import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _extract_post_id(post_url: str) -> str:
    m = re.search(r"/p/(\d+)", post_url)
    if not m:
        raise RuntimeError(f"无法从 URL 提取 post_id：{post_url}")
    return m.group(1)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_details_csv(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as fp:
        text = fp.read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return list(reader)


def _build_raw_from_rows(post_url: str, post_id: str, rows: List[Dict[str, str]], source_details_csv: str) -> Dict:
    title = ""
    floors = []
    for row in rows:
        if not title:
            title = (row.get("title") or "").strip()
        section = (row.get("section") or "").strip().lower()
        if section != "floor":
            continue
        try:
            idx = int(float((row.get("index") or "0").strip() or "0"))
        except Exception:
            idx = 0
        author = (row.get("author") or "").strip()
        time_str = (row.get("time") or "").strip()
        text = (row.get("text") or "").strip()
        images = (row.get("images") or "").strip()
        image_list = [s.strip() for s in images.split(";") if s.strip()] if images else []
        if not (text or image_list):
            continue
        floors.append(
            {
                "index": idx,
                "author": author,
                "time": time_str,
                "text": text,
                "images": image_list,
            }
        )
    floors = [f for f in floors if f.get("index", 0) > 0]
    floors.sort(key=lambda x: x.get("index", 0))
    return {
        "post_url": post_url,
        "post_id": post_id,
        "title": title,
        "floors": floors,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"details_csv": os.path.normpath(source_details_csv)},
    }


def import_project(post_url: str, details_csv_path: str, out_dir: str) -> Tuple[str, str]:
    post_id = _extract_post_id(post_url)
    rows = _read_details_csv(details_csv_path)
    if not rows:
        raise RuntimeError(f"details.csv 为空：{details_csv_path}")

    target_dir = os.path.join(out_dir, post_id, "source")
    _ensure_dir(target_dir)

    target_csv = os.path.join(target_dir, "details.csv")
    shutil.copyfile(details_csv_path, target_csv)

    raw_obj = _build_raw_from_rows(post_url=post_url, post_id=post_id, rows=rows, source_details_csv=details_csv_path)
    target_raw = os.path.join(target_dir, "raw.json")
    with open(target_raw, "w", encoding="utf-8") as fp:
        json.dump(raw_obj, fp, ensure_ascii=False, indent=2)

    return target_raw, target_csv


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--post-url", required=True)
    p.add_argument("--details-csv", required=True)
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "projects"))
    args = p.parse_args(argv)

    try:
        raw_path, csv_path = import_project(
            post_url=args.post_url.strip(),
            details_csv_path=args.details_csv,
            out_dir=args.out_dir,
        )
        print(json.dumps({"ok": True, "raw": raw_path, "csv": csv_path}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{e.__class__.__name__}: {e}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
