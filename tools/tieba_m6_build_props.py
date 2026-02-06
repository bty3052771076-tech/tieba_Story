import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def _load_manifest(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "render", "manifest.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到渲染 manifest：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict):
        raise RuntimeError("render/manifest.json 不是 JSON object")
    if not isinstance(obj.get("items"), list):
        raise RuntimeError("render/manifest.json 缺少 items 数组")
    return obj

def _load_storyboard(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "llm", "storyboard.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到分镜 storyboard：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict) or not isinstance(obj.get("scenes"), list):
        raise RuntimeError("storyboard.json 格式不正确：缺少 scenes 数组")
    return obj

def build_props(project_dir: str, public_prefix: str, max_scenes: int, bgm_rel: str, out_path: str) -> None:
    manifest = _load_manifest(project_dir)
    storyboard = _load_storyboard(project_dir)
    items: List[Dict[str, Any]] = []
    for it in manifest.get("items") or []:
        if isinstance(it, dict):
            items.append(it)
    if not items:
        raise RuntimeError("render/manifest.json.items 为空")
    items = items[: max(1, int(max_scenes))]

    prefix = "/" + public_prefix.strip("/ ")
    sb_scenes = storyboard.get("scenes") or []
    sb_by_id = {int(s.get("scene_id") or 0): s for s in sb_scenes if isinstance(s, dict)}
    scenes = []
    for it in items:
        sid = int(it.get("scene_id") or 0)
        if sid <= 0:
            continue
        dur = float(it.get("duration_s") or 0.0)
        sb = sb_by_id.get(sid) or {}
        subtitle = str(sb.get("voiceover_text") or "").strip()
        scenes.append(
            {
                "scene_id": sid,
                "image": f"{prefix}/images/scene_{sid:03d}.png",
                "audio": f"{prefix}/audio/scene_{sid:03d}.wav",
                "duration_s": dur,
                "subtitle": subtitle,
            }
        )
    if not scenes:
        raise RuntimeError("未能从 manifest 构造任何 scene")

    props = {
        "scenes": scenes,
        "bgm": {
            "src": "/" + bgm_rel.strip("/ "),
            "volume": 0.12,
        },
    }
    _write_json(out_path, props)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--public-prefix", required=True, help="public 下的相对前缀，例如 9461449190")
    p.add_argument("--bgm-rel", required=True, help="public 下 BGM 文件相对路径，例如 9461449190/bgm/bgm.mp3")
    p.add_argument("--max-scenes", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    build_props(
        project_dir=args.project_dir,
        public_prefix=args.public_prefix,
        max_scenes=int(args.max_scenes),
        bgm_rel=args.bgm_rel,
        out_path=args.out,
    )
    print(json.dumps({"ok": True, "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
