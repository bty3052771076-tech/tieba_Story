import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from prompt_pack import get_template, prompt_pack_version, render_user_prompt


@dataclass
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int


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


def _load_llm_config(config_path: str, model: str, temperature: float, max_tokens: int, timeout_s: int, base_url: str) -> LlmConfig:
    kv = _parse_simple_kv_config(_read_text(config_path)) if os.path.exists(config_path) else {}
    resolved_base = (base_url or kv.get("base_url") or "").strip()
    if not resolved_base:
        raise RuntimeError("缺少 base_url：请在 docs/aliyun_api-key.md 配置或用 --base-url 覆盖")
    api_key = (kv.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 api_key：请在 docs/aliyun_api-key.md 中配置（该文件应保持在 gitignore）")
    if "YOUR_" in api_key and "API_KEY" in api_key:
        raise RuntimeError("api_key 仍为占位符：请在 docs/aliyun_api-key.md 中填写真实 api_key")
    return LlmConfig(
        base_url=resolved_base.rstrip("/"),
        api_key=api_key,
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_s=int(timeout_s),
    )


def _chat_completions_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/v1") or b.endswith("/compatible-mode/v1"):
        return f"{b}/chat/completions"
    return f"{b}/compatible-mode/v1/chat/completions"


def _http_post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
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
        raise RuntimeError(f"LLM HTTPError: {e.code} {e.reason} {err_body}".strip())
    except TimeoutError as e:
        raise RuntimeError(f"LLM TimeoutError: {e}")
    except URLError as e:
        raise RuntimeError(f"LLM URLError: {e}")


def _extract_message_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"模型返回缺少 choices：{resp}")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"模型返回缺少 message.content：{resp}")
    return content


def _extract_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise RuntimeError("模型返回空文本")
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    raise RuntimeError("无法从模型返回中提取 JSON 对象")


def _load_storyboard(path: str) -> Dict[str, Any]:
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict) or not isinstance(obj.get("scenes"), list):
        raise RuntimeError("storyboard.json 格式不正确：缺少 scenes 数组")
    return obj


def _apply_voiceover_patch(storyboard: Dict[str, Any], patch: Dict[str, Any]) -> None:
    items = patch.get("scenes") or []
    if not isinstance(items, list):
        return
    by_id: Dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = int(it.get("scene_id") or 0)
        txt = str(it.get("voiceover_text") or "").strip()
        if sid > 0 and txt:
            by_id[sid] = txt
    for s in storyboard.get("scenes") or []:
        if not isinstance(s, dict):
            continue
        sid = int(s.get("scene_id") or 0)
        if sid in by_id:
            s["voiceover_text"] = by_id[sid]


def _apply_prompt_patch(storyboard: Dict[str, Any], patch: Dict[str, Any]) -> None:
    items = patch.get("scenes") or []
    if not isinstance(items, list):
        return
    by_id: Dict[int, Dict[str, str]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = int(it.get("scene_id") or 0)
        p = it.get("prompt") or {}
        if not isinstance(p, dict):
            continue
        pos = str(p.get("positive") or "").strip()
        neg = str(p.get("negative") or "").strip()
        if sid > 0 and pos and neg:
            by_id[sid] = {"positive": pos, "negative": neg}
    for s in storyboard.get("scenes") or []:
        if not isinstance(s, dict):
            continue
        sid = int(s.get("scene_id") or 0)
        if sid in by_id:
            s["prompt"] = by_id[sid]


def polish_storyboard(
    in_path: str,
    out_path: str,
    llm: LlmConfig,
    prompt_pack: str,
    mode: str,
    style_preset: str,
    persona: str,
) -> str:
    storyboard = _load_storyboard(in_path)
    url = _chat_completions_url(llm.base_url)

    modes = [m.strip() for m in mode.split(",") if m.strip()]
    if not modes:
        raise RuntimeError("mode 不能为空")

    for m in modes:
        if m == "voiceover":
            tpl_name = "voiceover_polish_v1"
        elif m == "prompts":
            tpl_name = "image_prompt_polish_v1"
        else:
            raise RuntimeError(f"不支持 mode：{m}（支持 voiceover,prompts 或用逗号组合）")

        tpl = get_template(prompt_pack, tpl_name)
        sys_prompt = tpl.system_prompt
        user_prompt = render_user_prompt(
            tpl.user_prompt_template,
            {"style_preset": style_preset, "persona": persona, "storyboard_json": storyboard},
        )
        payload: Dict[str, Any] = {
            "model": llm.model,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
        }

        content = ""
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = _http_post_json(url=url, api_key=llm.api_key, payload=payload, timeout_s=llm.timeout_s)
                content = _extract_message_content(resp)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 3:
                    time.sleep(1.2 * attempt)
        if last_err is not None:
            raise last_err

        patch = _extract_json_object(content)
        if m == "voiceover":
            _apply_voiceover_patch(storyboard, patch)
        elif m == "prompts":
            _apply_prompt_patch(storyboard, patch)

    meta = storyboard.get("meta") if isinstance(storyboard.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    meta["prompt_pack"] = prompt_pack_version(prompt_pack)
    meta["style_preset"] = style_preset
    meta["persona"] = persona
    meta["polished_at"] = datetime.now().isoformat(timespec="seconds")
    storyboard["meta"] = meta

    _write_json(out_path, storyboard)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="输入 storyboard.json 路径")
    p.add_argument("--out", dest="out_path", default="", help="输出路径（默认覆盖输入文件）")
    p.add_argument("--mode", default="voiceover,prompts", help="voiceover,prompts 或逗号组合")
    p.add_argument("--prompt-pack", default="v1")
    p.add_argument("--style-preset", default="documentary")
    p.add_argument("--persona", default="克制回望")
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"))
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="deepseek-v3.2")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args(argv)

    out_path = args.out_path or args.in_path
    llm = _load_llm_config(
        config_path=args.config,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        base_url=(args.base_url or "").strip(),
    )
    out = polish_storyboard(
        in_path=args.in_path,
        out_path=out_path,
        llm=llm,
        prompt_pack=str(args.prompt_pack),
        mode=str(args.mode),
        style_preset=str(args.style_preset),
        persona=str(args.persona),
    )
    print(json.dumps({"ok": True, "out": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

