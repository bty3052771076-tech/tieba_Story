import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fp:
        return fp.read()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: str) -> Any:
    return json.loads(_read_text(path))


def _must_keys(obj: Dict[str, Any], keys: List[str], label: str) -> None:
    for k in keys:
        if k not in obj:
            raise RuntimeError(f"{label} 缺少字段：{k}")


def _mp3_signature_ok(b: bytes) -> bool:
    if len(b) < 4:
        return False
    if b[0:3] == b"ID3":
        return True
    return b[0] == 0xFF and (b[1] & 0xE0) == 0xE0


def validate_project(project_dir: str) -> Tuple[bool, Dict[str, Any]]:
    plan_path = os.path.join(project_dir, "llm", "bgm.plan.json")
    bgm_manifest_path = os.path.join(project_dir, "assets", "bgm", "manifest.json")

    if not os.path.exists(plan_path):
        raise RuntimeError(f"缺少文件：{plan_path}")
    if not os.path.exists(bgm_manifest_path):
        raise RuntimeError(f"缺少文件：{bgm_manifest_path}")

    plan = _load_json(plan_path)
    if not isinstance(plan, dict):
        raise RuntimeError("bgm.plan.json 不是 JSON object")
    _must_keys(plan, ["style_keywords", "bpm_range", "mood", "mixing", "candidates", "selected", "risks"], "bgm.plan.json")

    selected = plan.get("selected") or {}
    if not isinstance(selected, dict):
        raise RuntimeError("bgm.plan.json.selected 不是对象")
    _must_keys(selected, ["title", "artist", "download_url", "license"], "bgm.plan.json.selected")

    candidates = plan.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("bgm.plan.json.candidates 为空")
    selected_key = (str(selected.get("title") or "").strip(), str(selected.get("artist") or "").strip())
    cand_keys = []
    for c in candidates:
        if isinstance(c, dict):
            cand_keys.append((str(c.get("title") or "").strip(), str(c.get("artist") or "").strip()))
    if selected_key not in cand_keys:
        raise RuntimeError("selected 不在 candidates 中（title+artist 不匹配）")

    bgm_manifest = _load_json(bgm_manifest_path)
    if not isinstance(bgm_manifest, dict):
        raise RuntimeError("assets/bgm/manifest.json 不是 JSON object")
    _must_keys(bgm_manifest, ["post_id", "post_url", "title", "artist", "download_url", "file", "license", "attribution"], "assets/bgm/manifest.json")

    f = bgm_manifest.get("file") or {}
    if not isinstance(f, dict):
        raise RuntimeError("assets/bgm/manifest.json.file 不是对象")
    _must_keys(f, ["path", "bytes", "sha256", "format"], "assets/bgm/manifest.json.file")

    audio_rel = str(f.get("path") or "")
    audio_path = os.path.join(project_dir, audio_rel)
    if not os.path.exists(audio_path):
        raise RuntimeError(f"缺少音频文件：{audio_path}")
    audio_bytes = _read_bytes(audio_path)
    if len(audio_bytes) != int(f.get("bytes")):
        raise RuntimeError(f"音频 bytes 不匹配：manifest={f.get('bytes')} 实际={len(audio_bytes)}")
    sha256 = _sha256_bytes(audio_bytes)
    if sha256 != str(f.get("sha256")):
        raise RuntimeError("音频 sha256 不匹配")
    if not _mp3_signature_ok(audio_bytes[:64]):
        raise RuntimeError("音频文件头不像 MP3（非 ID3/帧同步）")

    report = {
        "ok": True,
        "plan_path": plan_path,
        "bgm_manifest_path": bgm_manifest_path,
        "audio_path": audio_path,
        "selected": {
            "title": selected.get("title"),
            "artist": selected.get("artist"),
            "license": selected.get("license"),
            "download_url": selected.get("download_url"),
        },
        "audio": {
            "bytes": len(audio_bytes),
            "sha256": sha256,
        },
    }
    return True, report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    args = p.parse_args(argv)
    ok, report = validate_project(args.project_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

