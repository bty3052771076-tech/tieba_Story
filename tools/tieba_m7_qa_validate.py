import argparse
import json
import os
import re
import time
from dataclasses import dataclass
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


def _load_json(path: str) -> Any:
    return json.loads(_read_text(path))


def qa_validate(
    project_dir: str,
    llm: LlmConfig,
    prompt_pack: str,
    scene_count: int,
    target_total_seconds: int,
) -> str:
    summary_path = os.path.join(project_dir, "llm", "story.summary.json")
    storyboard_path = os.path.join(project_dir, "llm", "storyboard.json")
    bgm_path = os.path.join(project_dir, "llm", "bgm.plan.json")

    summary = _load_json(summary_path) if os.path.exists(summary_path) else {}
    storyboard = _load_json(storyboard_path) if os.path.exists(storyboard_path) else {}
    bgm = _load_json(bgm_path) if os.path.exists(bgm_path) else {}

    tpl = get_template(prompt_pack, "qa_validate_v1")
    sys_prompt = tpl.system_prompt
    user_prompt = render_user_prompt(
        tpl.user_prompt_template,
        {
            "scene_count": int(scene_count),
            "target_total_seconds": int(target_total_seconds),
            "story_summary_json": summary,
            "storyboard_json": storyboard,
            "bgm_plan_json": bgm,
        },
    )

    payload: Dict[str, Any] = {
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
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

    report = _extract_json_object(content)
    if not isinstance(report, dict):
        raise RuntimeError("QA 输出不是 JSON object")
    report["meta"] = {
        "prompt_pack": prompt_pack_version(prompt_pack),
        "scene_count": int(scene_count),
        "target_total_seconds": int(target_total_seconds),
    }

    out_path = os.path.join(project_dir, "llm", "qa.report.json")
    _write_json(out_path, report)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"))
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="deepseek-v3.2")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=2200)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--prompt-pack", default="v1")
    p.add_argument("--scenes", type=int, default=10)
    p.add_argument("--target-seconds", type=int, default=120)
    args = p.parse_args(argv)

    llm = _load_llm_config(
        config_path=args.config,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        base_url=(args.base_url or "").strip(),
    )
    out = qa_validate(
        project_dir=args.project_dir,
        llm=llm,
        prompt_pack=str(args.prompt_pack),
        scene_count=int(args.scenes),
        target_total_seconds=int(args.target_seconds),
    )
    print(json.dumps({"ok": True, "out": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

