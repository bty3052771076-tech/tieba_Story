import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from claude_cli import claude_print
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
    if not isinstance(obj.get("paragraphs"), list) or not isinstance(obj.get("trace"), list):
        raise RuntimeError("story.summary.json 缺少 paragraphs/trace 字段")
    return obj


def _validate_storyboard(storyboard: Dict[str, Any], scene_count: int, target_total_seconds: int, duration_tolerance_s: int) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list):
        return False, ["scenes 字段缺失或非数组"]
    if len(scenes) != int(scene_count):
        reasons.append(f"镜头数量不匹配：{len(scenes)} != {int(scene_count)}")
    total_duration = 0
    for i, s in enumerate(scenes, start=1):
        if not isinstance(s, dict):
            reasons.append(f"scene[{i}] 非对象")
            continue
        if not str(s.get("voiceover_text") or "").strip():
            reasons.append(f"scene[{i}] 缺少 voiceover_text")
        try:
            total_duration += int(s.get("duration_s") or 0)
        except Exception:
            reasons.append(f"scene[{i}] duration_s 非整数")
    if total_duration and abs(int(total_duration) - int(target_total_seconds)) > int(duration_tolerance_s):
        reasons.append(f"总时长异常：{total_duration} 秒（期望 {int(target_total_seconds)}±{int(duration_tolerance_s)} 秒）")
    for i, s in enumerate(scenes, start=1):
        trace = (s or {}).get("trace") or {}
        if not isinstance(trace, dict):
            reasons.append(f"scene[{i}] trace 缺失或非对象")
            continue
        if not isinstance(trace.get("paragraph_ids"), list) or not trace.get("paragraph_ids"):
            reasons.append(f"scene[{i}] trace.paragraph_ids 缺失或为空")
        floors = trace.get("floors") or {}
        if not isinstance(floors, dict):
            reasons.append(f"scene[{i}] trace.floors 缺失或非对象")
            continue
        if floors.get("start") is None or floors.get("end") is None:
            reasons.append(f"scene[{i}] trace.floors 缺少 start/end")
    return (len(reasons) == 0), reasons


def _autofill_trace(storyboard_obj: Dict[str, Any], summary: Dict[str, Any]) -> None:
    scenes = storyboard_obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return
    summary_trace = summary.get("trace")
    para_map: Dict[int, Dict[str, Any]] = {}
    if isinstance(summary_trace, list):
        for t in summary_trace:
            if not isinstance(t, dict):
                continue
            pid = int(t.get("paragraph_id") or 0)
            floors = t.get("floors") or {}
            if pid > 0 and isinstance(floors, dict) and floors.get("start") is not None and floors.get("end") is not None:
                para_map[pid] = {
                    "start": int(floors.get("start") or 0),
                    "end": int(floors.get("end") or 0),
                    "indexes": floors.get("indexes") if isinstance(floors.get("indexes"), list) else [],
                }
    para_count = max(1, len(para_map)) if para_map else 1
    for s in scenes:
        if not isinstance(s, dict):
            continue
        sid = int(s.get("scene_id") or 0)
        trace = s.get("trace") if isinstance(s.get("trace"), dict) else {}
        if not isinstance(trace, dict):
            trace = {}
        pids = trace.get("paragraph_ids")
        if not isinstance(pids, list) or not pids:
            pid = ((max(1, sid) - 1) % para_count) + 1
            trace["paragraph_ids"] = [pid]
        floors = trace.get("floors")
        if not isinstance(floors, dict) or floors.get("start") is None or floors.get("end") is None:
            pid0 = int((trace.get("paragraph_ids") or [1])[0] or 1)
            floors0 = para_map.get(pid0) or {"start": 0, "end": 0, "indexes": []}
            trace["floors"] = floors0
        if not isinstance((trace.get("floors") or {}).get("indexes"), list):
            trace["floors"]["indexes"] = []
        s["trace"] = trace


def generate_storyboard(
    project_dir: str,
    llm: LlmConfig,
    chars_per_second: float,
    scenes_count: int,
    target_total_seconds: int,
    prompt_pack: str,
    style_preset: str,
    persona: str,
    force: bool,
    provider: str,
) -> str:
    summary = _load_story_summary(project_dir)
    input_hash = _sha256_json(summary)

    out_path = os.path.join(project_dir, "llm", "storyboard.json")
    if not force and os.path.exists(out_path):
        try:
            prev = json.loads(_read_text(out_path))
            if isinstance(prev, dict) and prev.get("input_hash") == input_hash:
                prev_meta = prev.get("meta") or {}
                prev_llm = prev_meta.get("llm") or {}
                if str(prev_llm.get("model") or "") == llm.model and float(prev_meta.get("chars_per_second") or 0) == float(chars_per_second):
                    return out_path
        except Exception:
            pass

    url = _chat_completions_url(llm.base_url) if provider != "claude" else ""
    cps = max(1.0, float(chars_per_second))
    scene_count = max(8, int(scenes_count))
    target_s = max(30, int(target_total_seconds))
    duration_tolerance_s = 10

    tpl = get_template(prompt_pack, "storyboard_v3")
    sys_prompt = tpl.system_prompt
    user_prompt = render_user_prompt(
        tpl.user_prompt_template,
        {
            "scene_count": scene_count,
            "target_total_seconds": target_s,
            "chars_per_second": cps,
            "style_preset": style_preset,
            "persona": persona,
            "story_summary_json": summary,
        },
    )

    payload: Dict[str, Any] = {
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    content = ""
    resp: Optional[Dict[str, Any]] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            if provider == "claude":
                content = claude_print(user_prompt=user_prompt, system_prompt=sys_prompt)
            else:
                resp = _http_post_json(url=url, api_key=llm.api_key, payload=payload, timeout_s=llm.timeout_s)
                content = _extract_message_content(resp)
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(1.2 * attempt)
    if last_err:
        raise last_err

    result_obj: Dict[str, Any] = {}
    parse_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            result_obj = _extract_json_object(content)
            parse_err = None
            break
        except Exception as e:
            parse_err = e
            repair_sys = (
                sys_prompt
                + "\n\n"
                + "你上一次输出无法被严格 JSON 解析。请重新输出严格 JSON，并遵守：单行JSON、不得有真实换行符、不得有尾随逗号。"
            )
            repair_payload = {**payload, "temperature": 0.0, "messages": [{"role": "system", "content": repair_sys}, {"role": "user", "content": user_prompt}]}
            if provider == "claude":
                content = claude_print(user_prompt=user_prompt, system_prompt=repair_sys)
            else:
                resp_fix = _http_post_json(url=url, api_key=llm.api_key, payload=repair_payload, timeout_s=llm.timeout_s)
                content = _extract_message_content(resp_fix)
            time.sleep(0.4)
    if parse_err is not None:
        raise parse_err

    _autofill_trace(result_obj, summary)
    ok, reasons = _validate_storyboard(result_obj, scene_count=scene_count, target_total_seconds=target_s, duration_tolerance_s=duration_tolerance_s)
    if not ok:
        repair_sys = (
            sys_prompt
            + "\n\n"
            + "你上一次输出未满足结构/验收要求，请直接重写完整 JSON。\n"
            + "以下是不满足原因：\n"
            + "\n".join(f"- {r}" for r in reasons)
        )
        repair_payload = {**payload, "temperature": 0.1, "messages": [{"role": "system", "content": repair_sys}, {"role": "user", "content": user_prompt}]}
        if provider == "claude":
            content_fix = claude_print(user_prompt=user_prompt, system_prompt=repair_sys)
        else:
            resp_fix = _http_post_json(url=url, api_key=llm.api_key, payload=repair_payload, timeout_s=llm.timeout_s)
            content_fix = _extract_message_content(resp_fix)
        result_obj = _extract_json_object(content_fix)
        ok2, reasons2 = _validate_storyboard(result_obj, scene_count=scene_count, target_total_seconds=target_s, duration_tolerance_s=duration_tolerance_s)
        if not ok2:
            raise RuntimeError("分镜输出未通过校验：" + "; ".join(reasons2))

    _autofill_trace(result_obj, summary)
    out_obj = {
        "post_url": summary.get("post_url"),
        "post_id": summary.get("post_id"),
        "title": summary.get("title"),
        "synopsis": summary.get("synopsis"),
        "style_preset": style_preset,
        "persona": persona,
        "scenes": result_obj.get("scenes") or [],
        "input_hash": input_hash,
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "chars_per_second": cps,
            "target_total_seconds": target_s,
            "scene_count": scene_count,
            "prompt_pack": prompt_pack_version(prompt_pack),
            "llm": {"base_url": llm.base_url, "model": llm.model, "temperature": llm.temperature, "max_tokens": llm.max_tokens},
        },
    }
    _write_json(out_path, out_obj)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True, help="projects/{post_id} 目录（包含 llm/story.summary.json）")
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"), help="本地密钥配置文件（默认 docs/aliyun_api-key.md）")
    p.add_argument("--base-url", default="", help="可选：覆盖 base_url")
    p.add_argument("--model", default="glm-4.7", help="用于 M3 分镜生成的 LLM 模型名（OpenAI 兼容）")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=3500)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--chars-per-second", type=float, default=3.5, help="配音语速（字/秒），用于时长估算与约束提示")
    p.add_argument("--scenes", type=int, default=10, help="镜头数量（>=8；建议 10 便于抽检 10 条）")
    p.add_argument("--target-seconds", type=int, default=120, help="目标总时长（秒）")
    p.add_argument("--prompt-pack", default="v1", help="提示词模板包版本（prompts/packs/{pack}/prompt_pack.json）")
    p.add_argument("--style-preset", default="documentary", help="风格预设（用于提示词）")
    p.add_argument("--persona", default="克制回望", help="旁白人设（用于提示词）")
    p.add_argument("--provider", choices=["aliyun", "claude"], default="aliyun")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    provider = str(args.provider)
    if provider == "claude":
        llm = LlmConfig(
            base_url="claude",
            api_key="",
            model=str(args.model),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_s=int(args.timeout),
        )
    else:
        llm = _load_llm_config(
            config_path=args.config,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            base_url=(args.base_url or "").strip(),
        )
    out = generate_storyboard(
        project_dir=args.project_dir,
        llm=llm,
        chars_per_second=float(args.chars_per_second),
        scenes_count=max(8, int(args.scenes)),
        target_total_seconds=int(args.target_seconds),
        prompt_pack=str(args.prompt_pack),
        style_preset=str(args.style_preset),
        persona=str(args.persona),
        force=bool(args.force),
        provider=provider,
    )
    print(json.dumps({"ok": True, "out": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
