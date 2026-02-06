import argparse
import hashlib
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


def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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


def _load_story_summary(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "llm", "story.summary.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到 M2 产物：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict):
        raise RuntimeError("story.summary.json 格式不正确：期望 JSON object")
    return obj


def _load_storyboard(project_dir: str) -> Dict[str, Any]:
    path = os.path.join(project_dir, "llm", "storyboard.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到 M3 产物：{path}")
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict) or not isinstance(obj.get("scenes"), list):
        raise RuntimeError("storyboard.json 格式不正确：缺少 scenes 数组")
    return obj


def generate_bgm_plan(project_dir: str, llm: LlmConfig, prompt_pack: str, style_preset: str, persona: str, force: bool) -> str:
    summary = _load_story_summary(project_dir)
    storyboard = _load_storyboard(project_dir)
    input_hash = _sha256_json({"summary": summary, "storyboard": storyboard})
    out_path = os.path.join(project_dir, "llm", "bgm.plan.json")
    if not force and os.path.exists(out_path):
        try:
            prev = json.loads(_read_text(out_path))
            if isinstance(prev, dict) and prev.get("meta", {}).get("input_hash") == input_hash:
                return out_path
        except Exception:
            pass

    library = [
        {
            "title": "Porch Swing Days - slower",
            "artist": "Kevin MacLeod",
            "source": "incompetech",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Porch%20Swing%20Days%20-%20slower.mp3",
            "tags": ["nostalgic", "warm", "piano", "gentle"],
        },
        {
            "title": "On the Passing of Time",
            "artist": "Kevin MacLeod",
            "source": "incompetech",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/On%20the%20Passing%20of%20Time.mp3",
            "tags": ["reflective", "piano", "slow", "emotional"],
        },
        {
            "title": "Reminiscing",
            "artist": "Kevin MacLeod",
            "source": "incompetech",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Reminiscing.mp3",
            "tags": ["memory", "calm", "piano", "soft"],
        },
        {
            "title": "Windswept",
            "artist": "Kevin MacLeod",
            "source": "incompetech",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Windswept.mp3",
            "tags": ["wistful", "ambient", "soft", "wide"],
        },
        {
            "title": "With Regards",
            "artist": "Kevin MacLeod",
            "source": "incompetech",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/With%20Regards.mp3",
            "tags": ["romantic", "piano", "gentle", "warm"],
        },
    ]

    tpl = get_template(prompt_pack, "bgm_plan_v2")
    sys_prompt = tpl.system_prompt
    user_prompt = render_user_prompt(
        tpl.user_prompt_template,
        {
            "style_preset": style_preset,
            "persona": persona,
            "story_summary_json": summary,
            "storyboard_json": storyboard,
            "bgm_library_json": library,
        },
    )
    payload = {
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "response_format": {"type": "json_object"},
    }

    url = _chat_completions_url(llm.base_url)
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

    plan_obj = _extract_json_object(content)
    if not isinstance(plan_obj, dict):
        raise RuntimeError("模型输出不是 JSON object")

    candidates = plan_obj.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    selected = plan_obj.get("selected")
    if not isinstance(selected, dict):
        selected = {}

    def _find_in_library(t: str, a: str) -> Optional[Dict[str, Any]]:
        key_t = (t or "").strip().lower()
        key_a = (a or "").strip().lower()
        for item in library:
            it = (item.get("title") or "").strip().lower()
            ia = (item.get("artist") or "").strip().lower()
            if it == key_t and (not key_a or ia == key_a):
                return item
        for item in library:
            it = (item.get("title") or "").strip().lower()
            if it == key_t:
                return item
        return None

    if isinstance(selected, dict):
        st = str(selected.get("title") or "").strip()
        sa = str(selected.get("artist") or "").strip()
        ref = _find_in_library(st, sa)
        if ref is not None:
            selected["download_url"] = ref.get("download_url")
            selected["license"] = ref.get("license")
            selected["license_url"] = ref.get("license_url")
            selected["source"] = ref.get("source")
            selected["artist"] = ref.get("artist") or selected.get("artist")
            selected["title"] = ref.get("title") or selected.get("title")
        plan_obj["selected"] = selected

    out_obj = {
        "post_url": summary.get("post_url"),
        "post_id": summary.get("post_id"),
        "title": summary.get("title"),
        "synopsis": summary.get("synopsis"),
        **plan_obj,
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_hash": input_hash,
            "prompt_pack": prompt_pack_version(prompt_pack),
            "style_preset": style_preset,
            "persona": persona,
            "llm": {
                "base_url": llm.base_url,
                "model": llm.model,
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
            },
        },
    }
    _write_json(out_path, out_obj)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"))
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="deepseek-v3.2")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2200)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--prompt-pack", default="v1")
    p.add_argument("--style-preset", default="documentary")
    p.add_argument("--persona", default="克制回望")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    llm = _load_llm_config(
        config_path=args.config,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        base_url=(args.base_url or "").strip(),
    )
    out = generate_bgm_plan(
        project_dir=args.project_dir,
        llm=llm,
        prompt_pack=str(args.prompt_pack),
        style_preset=str(args.style_preset),
        persona=str(args.persona),
        force=bool(args.force),
    )
    print(json.dumps({"ok": True, "out": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
