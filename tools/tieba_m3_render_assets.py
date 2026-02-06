import argparse
import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Config:
    api_key: str
    base_url: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def _parse_simple_kv_config(text: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(.*)"\s*$', s)
        if not m:
            continue
        kv[m.group(1)] = m.group(2)
    return kv


def _load_config(config_path: str, base_url: str) -> Config:
    kv = _parse_simple_kv_config(_read_text(config_path)) if os.path.exists(config_path) else {}
    resolved_base = (base_url or kv.get("base_url") or "").strip()
    if not resolved_base:
        raise RuntimeError("缺少 base_url：请在 docs/aliyun_api-key.md 配置或用 --base-url 覆盖")
    api_key = (kv.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 api_key：请在 docs/aliyun_api-key.md 中配置（该文件应保持在 gitignore）")
    if "YOUR_" in api_key and "API_KEY" in api_key:
        raise RuntimeError("api_key 仍为占位符：请在 docs/aliyun_api-key.md 中填写真实 api_key")
    return Config(api_key=api_key, base_url=resolved_base.rstrip("/"))


def _http_post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTPError: {e.code} {e.reason} {err_body}".strip())
    except TimeoutError as e:
        raise RuntimeError(f"TimeoutError: {e}")
    except URLError as e:
        raise RuntimeError(f"URLError: {e}")


def _http_get_bytes(url: str, timeout_s: int) -> bytes:
    if isinstance(url, str) and url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            req = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Connection": "close"},
                method="GET",
            )
            with urlopen(req, timeout=timeout_s) as resp:
                chunks: List[bytes] = []
                while True:
                    buf = resp.read(1024 * 64)
                    if not buf:
                        break
                    chunks.append(buf)
                return b"".join(chunks)
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(1.0 * attempt)
                continue
            raise
    raise last_err or RuntimeError("下载失败")


def _images_generations_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/api/v1"):
        return f"{b}/services/aigc/multimodal-generation/generation"
    return f"{b}/api/v1/services/aigc/multimodal-generation/generation"


def _tts_generation_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/api/v1"):
        return f"{b}/services/aigc/multimodal-generation/generation"
    return f"{b}/api/v1/services/aigc/multimodal-generation/generation"


def _load_storyboard(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "llm", "storyboard.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到 M3 产物：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict) or not isinstance(obj.get("scenes"), list):
        raise RuntimeError("storyboard.json 格式不正确：缺少 scenes 数组")
    return obj


def _load_existing_manifest(project_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(project_dir, "render", "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        obj = json.loads(_read_text(path))
        if not isinstance(obj, dict):
            return None
        if not isinstance(obj.get("items"), list):
            return None
        return obj
    except Exception:
        return None


def _save_png_from_b64(b64_data: str, out_path: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    img_bytes = base64.b64decode(b64_data)
    with open(out_path, "wb") as fp:
        fp.write(img_bytes)


def _encode_image_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    with open(file_path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"



def generate_scene_image(
    cfg: Config,
    model: str,
    prompt: str,
    negative_prompt: str,
    size: str,
    prompt_extend: bool,
    seed: Optional[int],
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    url = _images_generations_url(cfg.base_url)
    parameters: Dict[str, Any] = {
        "watermark": False,
        "prompt_extend": bool(prompt_extend),
        "size": size,
    }
    if negative_prompt.strip():
        parameters["negative_prompt"] = negative_prompt.strip()
    if seed is not None:
        parameters["seed"] = int(seed)

    payload: Dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": parameters,
    }

    resp = _http_post_json(url=url, api_key=cfg.api_key, payload=payload, timeout_s=timeout_s)
    output = resp.get("output") or {}
    choices = output.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"生图返回缺少 output.choices：{resp}")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content") or []
    if not isinstance(content, list) or not content:
        raise RuntimeError(f"生图返回缺少 message.content：{resp}")
    image_url = (content[0] or {}).get("image")
    if not isinstance(image_url, str) or not image_url.strip():
        raise RuntimeError(f"生图返回缺少 image url：{resp}")
    return image_url, resp


def edit_scene_image(
    cfg: Config,
    model: str,
    base_image_data_url: str,
    instruction_text: str,
    negative_prompt: str,
    size: str,
    seed: Optional[int],
    prompt_extend: bool,
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    url = _images_generations_url(cfg.base_url)
    parameters: Dict[str, Any] = {
        "watermark": False,
        "size": size,
        "prompt_extend": bool(prompt_extend),
    }
    if negative_prompt.strip():
        parameters["negative_prompt"] = negative_prompt.strip()
    if seed is not None:
        parameters["seed"] = int(seed)

    payload: Dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": base_image_data_url},
                        {"text": instruction_text},
                    ],
                }
            ]
        },
        "parameters": parameters,
    }

    resp = _http_post_json(url=url, api_key=cfg.api_key, payload=payload, timeout_s=timeout_s)
    output = resp.get("output") or {}
    choices = output.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"生图返回缺少 output.choices：{resp}")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content") or []
    if not isinstance(content, list) or not content:
        raise RuntimeError(f"生图返回缺少 message.content：{resp}")
    image_url = (content[0] or {}).get("image")
    if not isinstance(image_url, str) or not image_url.strip():
        raise RuntimeError(f"生图返回缺少 image url：{resp}")
    return image_url, resp


def generate_scene_tts(
    cfg: Config,
    model: str,
    text: str,
    voice: str,
    language_type: str,
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    url = _tts_generation_url(cfg.base_url)
    payload: Dict[str, Any] = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "language_type": language_type,
        },
    }
    resp = _http_post_json(url=url, api_key=cfg.api_key, payload=payload, timeout_s=timeout_s)
    output = resp.get("output") or {}
    audio = (output.get("audio") or {}) if isinstance(output, dict) else {}
    audio_url = audio.get("url")
    if not isinstance(audio_url, str) or not audio_url.strip():
        raise RuntimeError(f"TTS 返回缺少 output.audio.url：{resp}")
    return audio_url, resp


def render_assets(
    project_dir: str,
    cfg: Config,
    image_model: str,
    edit_model: str,
    consistency_mode: str,
    prompt_extend: bool,
    vary_seed: bool,
    tts_model: str,
    tts_voice: str,
    tts_language_type: str,
    size: str,
    seed: Optional[int],
    max_scenes: int,
    only_tts: bool,
    timeout_s: int,
    sleep_s: float,
) -> str:
    storyboard = _load_storyboard(project_dir)
    scenes: List[Dict[str, Any]] = storyboard.get("scenes") or []
    post_id = str(storyboard.get("post_id") or os.path.basename(project_dir))

    out_root = os.path.join(project_dir, "render")
    img_dir = os.path.join(out_root, "images")
    aud_dir = os.path.join(out_root, "audio")
    _ensure_dir(img_dir)
    _ensure_dir(aud_dir)

    prev_manifest = _load_existing_manifest(project_dir)
    existing_by_scene: Dict[int, Dict[str, Any]] = {}
    if prev_manifest is not None:
        for it in prev_manifest.get("items") or []:
            if not isinstance(it, dict):
                continue
            try:
                sid = int(it.get("scene_id"))
            except Exception:
                continue
            existing_by_scene[sid] = it

    manifest: Dict[str, Any] = {
        "post_id": post_id,
        "post_url": storyboard.get("post_url"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_model": image_model,
        "edit_model": edit_model,
        "consistency_mode": consistency_mode,
        "prompt_extend": bool(prompt_extend),
        "vary_seed": bool(vary_seed),
        "tts_model": tts_model,
        "tts_voice": tts_voice,
        "tts_language_type": tts_language_type,
        "size": size,
        "seed": seed,
        "only_tts": bool(only_tts),
        "items": [],
    }

    prev_img_path: Optional[str] = None

    for idx, scene in enumerate(scenes[: max(1, int(max_scenes))], start=1):
        scene_id = int(scene.get("scene_id") or idx)
        voiceover = str(scene.get("voiceover_text") or "").strip()
        prompt_obj = scene.get("prompt") or {}
        positive = str((prompt_obj.get("positive") if isinstance(prompt_obj, dict) else "") or "").strip()
        negative = str((prompt_obj.get("negative") if isinstance(prompt_obj, dict) else "") or "").strip()

        img_path = os.path.join(img_dir, f"scene_{scene_id:03d}.png")
        aud_path = os.path.join(aud_dir, f"scene_{scene_id:03d}.wav")

        seed_for_scene: Optional[int] = seed
        if vary_seed and seed is not None:
            seed_for_scene = int(seed) + int(scene_id)

        img_method = "text2image"
        img_url = ""
        img_resp: Dict[str, Any] = {}
        if not only_tts:
            if consistency_mode == "edit_prev" and prev_img_path and os.path.exists(prev_img_path):
                base_data_url = _encode_image_to_data_url(prev_img_path)
                visual_description = str(scene.get("visual_description") or "").strip()
                composition = str(scene.get("composition") or "").strip()
                mood = str(scene.get("mood") or "").strip()
                instruction_text = (
                    "请保持与 Image 1 完全一致的画风、色调、光线与人物身份（脸部、发型、年龄感），"
                    "在此基础上修改场景与动作，使其符合以下描述。不要添加任何文字、水印或Logo。\n"
                    f"场景描述：{visual_description}\n"
                    f"构图建议：{composition}\n"
                    f"氛围：{mood}\n"
                ).strip()
                img_url, img_resp = edit_scene_image(
                    cfg=cfg,
                    model=edit_model,
                    base_image_data_url=base_data_url,
                    instruction_text=instruction_text,
                    negative_prompt=negative,
                    size=size,
                    seed=seed_for_scene,
                    prompt_extend=prompt_extend,
                    timeout_s=timeout_s,
                )
                img_method = "edit_prev"
            else:
                img_url, img_resp = generate_scene_image(
                    cfg=cfg,
                    model=image_model,
                    prompt=positive,
                    negative_prompt=negative,
                    size=size,
                    seed=seed_for_scene,
                    prompt_extend=prompt_extend,
                    timeout_s=timeout_s,
                )
            img_bytes = _http_get_bytes(img_url, timeout_s=timeout_s)
            with open(img_path, "wb") as fp:
                fp.write(img_bytes)
            time.sleep(max(0.0, float(sleep_s)))
        else:
            prev = existing_by_scene.get(scene_id) or {}
            prev_image_path = str(prev.get("image_path") or "").strip()
            if prev_image_path:
                img_path = os.path.join(project_dir, prev_image_path)
            img_url = str(prev.get("image_url") or "").strip()
            img_method = str(prev.get("image_method") or "existing")

        audio_url, tts_resp = generate_scene_tts(
            cfg=cfg,
            model=tts_model,
            text=voiceover,
            voice=tts_voice,
            language_type=tts_language_type,
            timeout_s=timeout_s,
        )
        wav_bytes = _http_get_bytes(audio_url, timeout_s=timeout_s)
        with open(aud_path, "wb") as fp:
            fp.write(wav_bytes)
        time.sleep(max(0.0, float(sleep_s)))

        manifest["items"].append(
            {
                "scene_id": scene_id,
                "duration_s": scene.get("duration_s"),
                "image_path": os.path.relpath(img_path, project_dir) if img_path else "",
                "audio_path": os.path.relpath(aud_path, project_dir),
                "image_url": img_url,
                "tts_audio_url": audio_url,
                "trace": scene.get("trace"),
                "image_method": img_method,
                "image_response_meta": {"request_id": img_resp.get("request_id"), "usage": img_resp.get("usage")} if img_resp else {},
                "tts_response_meta": {"request_id": tts_resp.get("request_id"), "output": (tts_resp.get("output") or {})},
            }
        )
        prev_img_path = img_path

    manifest_path = os.path.join(out_root, "manifest.json")
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"))
    p.add_argument("--base-url", default="", help="可选：覆盖 base_url（默认读 config）")
    p.add_argument("--image-model", default="qwen-image-max")
    p.add_argument("--edit-model", default="qwen-image-edit-plus-2025-12-15")
    p.add_argument("--size", default="1664*928")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--consistency-mode", default="edit_prev", choices=["prompt", "edit_prev"])
    p.add_argument("--prompt-extend", action="store_true", help="启用 prompt_extend（默认关闭以增强一致性）")
    p.add_argument("--vary-seed", action="store_true", help="每镜头 seed=base+scene_id（默认关闭以增强一致性）")
    p.add_argument("--tts-model", default="qwen3-tts-flash")
    p.add_argument("--tts-voice", default="Neil")
    p.add_argument("--tts-language-type", default="Chinese")
    p.add_argument("--max-scenes", type=int, default=2)
    p.add_argument("--only-tts", action="store_true", help="只重跑配音（复用已有图片/manifest）")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--sleep", type=float, default=0.8)
    args = p.parse_args()

    cfg = _load_config(config_path=args.config, base_url=(args.base_url or "").strip())
    manifest_path = render_assets(
        project_dir=args.project_dir,
        cfg=cfg,
        image_model=args.image_model,
        edit_model=args.edit_model,
        consistency_mode=str(args.consistency_mode),
        prompt_extend=bool(args.prompt_extend),
        vary_seed=bool(args.vary_seed),
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_language_type=args.tts_language_type,
        size=args.size,
        seed=int(args.seed) if args.seed is not None else None,
        max_scenes=max(1, int(args.max_scenes)),
        only_tts=bool(args.only_tts),
        timeout_s=int(args.timeout),
        sleep_s=float(args.sleep),
    )
    print(json.dumps({"ok": True, "manifest": manifest_path}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
