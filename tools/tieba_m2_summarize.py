import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
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


@dataclass
class VlmConfig:
    base_url: str
    api_key: str
    model: str
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
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _parse_simple_kv_config(text: str) -> Dict[str, str]:
    """
    读取类似 docs/aliyun_api-key.md 的简易 key="value" 配置。
    只解析最底部形如：key="value" 的行，忽略注释与空行。
    """
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


def _is_placeholder_api_key(api_key: str) -> bool:
    k = (api_key or "").strip()
    if not k:
        return True
    placeholders = {
        "YOUR_ALIYUN_BAILIAN_API_KEY",
        "YOUR_DASHSCOPE_API_KEY",
        "YOUR_OPENAI_API_KEY",
        "YOUR_API_KEY",
    }
    if k in placeholders:
        return True
    if "YOUR_" in k and "API_KEY" in k:
        return True
    return False


def _load_llm_config(
    config_path: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    base_url: Optional[str],
) -> LlmConfig:
    """
    优先从本地配置文件读取 base_url/api_key（该文件应被 gitignore）。
    若未提供配置文件或字段缺失，则从环境变量读取 api_key。
    """
    kv: Dict[str, str] = {}
    if config_path and os.path.exists(config_path):
        kv = _parse_simple_kv_config(_read_text(config_path))

    resolved_base = (base_url or kv.get("base_url") or "").strip()
    if not resolved_base:
        raise RuntimeError("缺少 base_url：请用 --base-url 或在配置文件中提供 base_url")

    api_key_candidates = [
        kv.get("api_key"),
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("ALIYUN_BAILIAN_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    ]
    resolved_key = next((k.strip() for k in api_key_candidates if k and k.strip()), "")
    if not resolved_key:
        raise RuntimeError(
            "缺少 api_key：请在配置文件中设置 api_key，或设置环境变量 DASHSCOPE_API_KEY/ALIYUN_BAILIAN_API_KEY/OPENAI_API_KEY"
        )
    if _is_placeholder_api_key(resolved_key):
        raise RuntimeError("api_key 仍是占位符：请在 docs/aliyun_api-key.md 中填写真实 api_key（该文件应保持在 gitignore）")

    return LlmConfig(
        base_url=resolved_base.rstrip("/"),
        api_key=resolved_key,
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_s=int(timeout_s),
    )


def _load_vlm_config(
    config_path: Optional[str],
    model: str,
    timeout_s: int,
    base_url: Optional[str],
) -> VlmConfig:
    kv: Dict[str, str] = {}
    if config_path and os.path.exists(config_path):
        kv = _parse_simple_kv_config(_read_text(config_path))

    resolved_base = (base_url or kv.get("base_url") or "").strip()
    if not resolved_base:
        raise RuntimeError("缺少 base_url：请用 --base-url 或在配置文件中提供 base_url")

    api_key_candidates = [
        kv.get("api_key"),
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("ALIYUN_BAILIAN_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    ]
    resolved_key = next((k.strip() for k in api_key_candidates if k and k.strip()), "")
    if not resolved_key:
        raise RuntimeError(
            "缺少 api_key：请在配置文件中设置 api_key，或设置环境变量 DASHSCOPE_API_KEY/ALIYUN_BAILIAN_API_KEY/OPENAI_API_KEY"
        )
    if _is_placeholder_api_key(resolved_key):
        raise RuntimeError("api_key 仍是占位符：请在 docs/aliyun_api-key.md 中填写真实 api_key（该文件应保持在 gitignore）")

    return VlmConfig(base_url=resolved_base.rstrip("/"), api_key=resolved_key, model=model, timeout_s=int(timeout_s))


def _extract_post_id_from_project_dir(project_dir: str) -> str:
    return os.path.basename(os.path.normpath(project_dir))


def _load_project_source(project_dir: str) -> Dict[str, Any]:
    raw_path = os.path.join(project_dir, "source", "raw.json")
    if not os.path.exists(raw_path):
        raise RuntimeError(f"未找到 raw.json：{raw_path}")
    raw = json.loads(_read_text(raw_path))
    if not isinstance(raw, dict):
        raise RuntimeError("raw.json 格式不正确：期望 JSON object")
    return raw


def _clean_floor_text(text: str) -> str:
    """
    轻量清洗：去尾巴、归一化空白。
    不做强规则删除，避免误删剧情内容。
    """
    t = (text or "").replace("\u200b", "")
    t = re.sub(r"\s+", " ", t).strip()
    tails = [
        "来自Android客户端",
        "来自iPhone客户端",
        "来自手机百度贴吧",
        "来自百度贴吧客户端",
        "来自贴吧",
    ]
    for tail in tails:
        if t.endswith(tail):
            t = t[: -len(tail)].strip()
    return t


def _build_m2_input(raw: Dict[str, Any], max_floors: int, max_chars: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    post_url = str(raw.get("post_url") or "").strip()
    title = str(raw.get("title") or "").strip()
    floors = raw.get("floors") or []
    if not isinstance(floors, list):
        raise RuntimeError("raw.json floors 字段格式不正确：期望 list")

    normalized: List[Dict[str, Any]] = []
    total_chars = 0
    for f in floors:
        if not isinstance(f, dict):
            continue
        idx = int(f.get("index") or 0)
        author = str(f.get("author") or "")
        time_str = str(f.get("time") or "")
        text = _clean_floor_text(str(f.get("text") or ""))
        images = f.get("images") or []
        if not isinstance(images, list):
            images = []
        images = [str(x).strip() for x in images if str(x).strip()]

        if not (text or images):
            continue

        entry = {"index": idx, "author": author, "time": time_str, "text": text, "images": images}
        normalized.append(entry)
        total_chars += len(text) + sum(len(u) for u in images)
        if len(normalized) >= max_floors or total_chars >= max_chars:
            break

    normalized = [x for x in normalized if x.get("index", 0) > 0]
    normalized.sort(key=lambda x: x.get("index", 0))
    if not normalized:
        raise RuntimeError("可用于总结的楼层为空（可能 raw.json 中 floors 为空）")

    meta = {
        "post_url": post_url,
        "title": title,
        "floor_min": normalized[0]["index"],
        "floor_max": normalized[-1]["index"],
        "floor_count": len(normalized),
    }
    return meta, normalized


def _chat_completions_url(base_url: str) -> str:
    """
    DashScope 的 OpenAI 兼容模式通常为 /compatible-mode/v1。
    若用户已经传入包含 /v1 或 /compatible-mode/v1 的 base_url，则直接拼 chat/completions。
    """
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


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    兼容模型偶发的“前后加解释文字”情况：尽量从文本中截出 JSON object。
    """
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


def _extract_message_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices or not isinstance(choices, list):
        raise RuntimeError(f"模型返回缺少 choices：{resp}")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"模型返回缺少 message.content：{resp}")
    return content


def _estimate_narration_constraints(
    min_paragraph_seconds: float,
    target_min_seconds: float,
    target_max_seconds: float,
    chars_per_second: float,
) -> Dict[str, Any]:
    cps = max(1.0, float(chars_per_second))
    min_para_s = max(1.0, float(min_paragraph_seconds))
    tgt_min_s = max(min_para_s, float(target_min_seconds))
    tgt_max_s = max(tgt_min_s, float(target_max_seconds))

    min_para_chars = int(round(min_para_s * cps))
    tgt_min_chars = int(round(tgt_min_s * cps))
    tgt_max_chars = int(round(tgt_max_s * cps))

    fixed_paras = int(tgt_min_s // min_para_s)
    if fixed_paras * min_para_s < tgt_min_s:
        fixed_paras += 1
    fixed_paras = max(1, fixed_paras)
    min_paras = fixed_paras
    max_paras = fixed_paras

    max_para_chars = int(max(min_para_chars, int(round(tgt_max_chars / max(1, fixed_paras)))))
    ideal_para_chars = int(min(max_para_chars, max(min_para_chars, int(round(((tgt_min_chars + tgt_max_chars) / 2.0) / max(1, fixed_paras))))))
    tgt_min_chars = max(tgt_min_chars, min_para_chars * fixed_paras)
    tgt_max_chars = min(tgt_max_chars, max_para_chars * fixed_paras)

    return {
        "min_paragraph_seconds": min_para_s,
        "target_min_seconds": tgt_min_s,
        "target_max_seconds": tgt_max_s,
        "chars_per_second": cps,
        "min_paragraph_chars": min_para_chars,
        "max_paragraph_chars": max_para_chars,
        "target_min_chars": tgt_min_chars,
        "target_max_chars": tgt_max_chars,
        "min_paragraphs": min_paras,
        "max_paragraphs": max_paras,
        "ideal_paragraph_chars": ideal_para_chars,
    }


def _check_narration_constraints(paragraph_texts: List[str], narration: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    min_paras = int(narration.get("min_paragraphs") or 0)
    max_paras = int(narration.get("max_paragraphs") or 0)
    min_chars = int(narration.get("min_paragraph_chars") or 0)
    max_chars = int(narration.get("max_paragraph_chars") or 10**9)
    tgt_min_chars = int(narration.get("target_min_chars") or 0)
    tgt_max_chars = int(narration.get("target_max_chars") or 10**9)

    n = len(paragraph_texts)
    if n < min_paras or n > max_paras:
        reasons.append(f"段落数量不在范围内：{n} 不在 {min_paras}–{max_paras}")

    lens = [len((t or "").strip()) for t in paragraph_texts]
    if lens:
        if min(lens) < min_chars:
            reasons.append(f"存在段落过短：min={min(lens)} < {min_chars}")
        if max(lens) > max_chars:
            reasons.append(f"存在段落过长：max={max(lens)} > {max_chars}")
        total = sum(lens)
        if total < tgt_min_chars or total > tgt_max_chars:
            reasons.append(f"总字数不在范围内：total={total} 不在 {tgt_min_chars}–{tgt_max_chars}")
    return (len(reasons) == 0), reasons


def _truncate_text_to_len(text: str, target_len: int) -> str:
    t = (text or "").strip()
    if target_len <= 0:
        return ""
    if len(t) <= target_len:
        return t
    candidates = ["。", "！", "？", "；", ".", "!", "?", ";", "…", "）", ")", "”", "\""]
    window = t[: target_len + 1]
    cut = -1
    for c in candidates:
        pos = window.rfind(c)
        if pos > cut:
            cut = pos
    if cut >= 20:
        return window[: cut + 1].strip()
    return t[:target_len].strip()


def _enforce_narration_by_truncation(paragraphs: List[Dict[str, Any]], narration: Dict[str, Any]) -> List[Dict[str, Any]]:
    min_chars = int(narration.get("min_paragraph_chars") or 0)
    max_chars = int(narration.get("max_paragraph_chars") or 10**9)
    tgt_max_chars = int(narration.get("target_max_chars") or 10**9)

    fixed: List[Dict[str, Any]] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        t = str(p.get("text") or "").strip()
        if len(t) > max_chars:
            t = _truncate_text_to_len(t, max_chars)
        fixed.append({**p, "text": t})

    texts = [str(x.get("text") or "").strip() for x in fixed]
    lens = [len(x) for x in texts]
    total = sum(lens)
    over = total - tgt_max_chars
    if over <= 0:
        return fixed

    for _ in range(200):
        lens = [len(str(x.get("text") or "").strip()) for x in fixed]
        total = sum(lens)
        over = total - tgt_max_chars
        if over <= 0:
            break
        idx = -1
        longest = -1
        for i, ln in enumerate(lens):
            if ln > longest and ln > min_chars:
                longest = ln
                idx = i
        if idx < 0:
            break
        reducible = longest - min_chars
        reduce_by = min(reducible, over)
        target_len = max(min_chars, longest - reduce_by)
        fixed[idx]["text"] = _truncate_text_to_len(str(fixed[idx].get("text") or ""), target_len)

    return fixed


def _load_image_desc_cache(project_dir: str) -> Dict[str, Any]:
    cache_path = os.path.join(project_dir, "llm", "vlm.image_descriptions.json")
    if not os.path.exists(cache_path):
        return {"items": {}}
    try:
        obj = json.loads(_read_text(cache_path))
        if isinstance(obj, dict) and isinstance(obj.get("items"), dict):
            return obj
    except Exception:
        pass
    return {"items": {}}


def _save_image_desc_cache(project_dir: str, cache_obj: Dict[str, Any]) -> None:
    cache_path = os.path.join(project_dir, "llm", "vlm.image_descriptions.json")
    _write_json(cache_path, cache_obj)


def _guess_mime_type_from_url(image_url: str, fallback: str = "image/jpeg") -> str:
    try:
        u = urlparse(image_url)
        path = u.path or ""
        mime, _ = mimetypes.guess_type(path)
        if mime and mime.startswith("image/"):
            return mime
    except Exception:
        pass
    return fallback


def _download_image_bytes(image_url: str, timeout_s: int, max_bytes: int = 8 * 1024 * 1024) -> Tuple[bytes, str]:
    req = Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Connection": "close",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ct.startswith("image/"):
            ct = _guess_mime_type_from_url(image_url)
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError(f"图片过大（>{max_bytes} bytes），无法发送给 VLM：{image_url}")
        return data, ct


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _describe_image(vlm: VlmConfig, image_url: str, floor_index: int, prompt_pack: str) -> Dict[str, Any]:
    url = _chat_completions_url(vlm.base_url)
    img_bytes, mime_type = _download_image_bytes(image_url=image_url, timeout_s=max(10, int(vlm.timeout_s)))
    data_url = _to_data_url(image_bytes=img_bytes, mime_type=mime_type)
    tpl = get_template(prompt_pack, "vlm_describe_v1")
    sys_prompt = tpl.system_prompt
    user_content = [
        {"type": "text", "text": render_user_prompt(tpl.user_prompt_template, {"floor_index": int(floor_index)})},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    payload: Dict[str, Any] = {
        "model": vlm.model,
        "temperature": 0.1,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    last_err: Optional[Exception] = None
    resp: Optional[Dict[str, Any]] = None
    for attempt in range(1, 4):
        try:
            resp = _http_post_json(url=url, api_key=vlm.api_key, payload=payload, timeout_s=vlm.timeout_s)
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(1.0 * attempt)
    if last_err:
        raise last_err
    if not isinstance(resp, dict):
        raise RuntimeError("VLM 返回格式不正确：期望 JSON object")
    content = _extract_message_content(resp)
    obj = _extract_json_object(content)
    caption = str(obj.get("caption") or "").strip()
    ocr_text = str(obj.get("ocr_text") or "").strip()
    entities = obj.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    entities = [str(x).strip() for x in entities if str(x).strip()]
    return {
        "image_url": image_url,
        "floor_index": int(floor_index),
        "caption": caption,
        "ocr_text": ocr_text,
        "entities": entities[:8],
        "model": vlm.model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _maybe_backup_offline_summary(project_dir: str, summary_path: str) -> Optional[str]:
    if not os.path.exists(summary_path):
        return None
    try:
        obj = json.loads(_read_text(summary_path))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    uncertainties = obj.get("uncertainties") or []
    if isinstance(uncertainties, list) and any("离线兜底输出" in str(x) for x in uncertainties):
        backup_path = os.path.join(project_dir, "llm", "story.summary.offline.json")
        if not os.path.exists(backup_path):
            _ensure_dir(os.path.dirname(backup_path))
            shutil.copyfile(summary_path, backup_path)
            return backup_path
    return None


def _coerce_trace(paragraphs: List[Dict[str, Any]], valid_floor_indexes: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将模型返回的 floors 引用尽量规整到真实存在的楼层范围内，保证可追溯性字段稳定。
    """
    valid_set = set(valid_floor_indexes)
    trace: List[Dict[str, Any]] = []
    fixed_paragraphs: List[Dict[str, Any]] = []
    for i, p in enumerate(paragraphs, start=1):
        if not isinstance(p, dict):
            continue
        text = str(p.get("text") or "").strip()
        floors = p.get("floors") or {}
        if not isinstance(floors, dict):
            floors = {}

        idxs = floors.get("indexes") or floors.get("index_list") or []
        if not isinstance(idxs, list):
            idxs = []
        idxs = [int(x) for x in idxs if str(x).strip().isdigit()]
        idxs = [x for x in idxs if x in valid_set]
        idxs = sorted(set(idxs))

        start = floors.get("start")
        end = floors.get("end")
        try:
            start_i = int(start)
        except Exception:
            start_i = idxs[0] if idxs else min(valid_floor_indexes)
        try:
            end_i = int(end)
        except Exception:
            end_i = idxs[-1] if idxs else max(valid_floor_indexes)

        if start_i not in valid_set and idxs:
            start_i = idxs[0]
        if end_i not in valid_set and idxs:
            end_i = idxs[-1]

        start_i = max(min(valid_floor_indexes), min(start_i, max(valid_floor_indexes)))
        end_i = max(min(valid_floor_indexes), min(end_i, max(valid_floor_indexes)))
        if start_i > end_i:
            start_i, end_i = end_i, start_i

        fixed_paragraphs.append({"paragraph_id": i, "text": text})
        trace.append({"paragraph_id": i, "floors": {"start": start_i, "end": end_i, "indexes": idxs}})
    return fixed_paragraphs, trace


def _summarize_offline(meta: Dict[str, Any], floors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    离线兜底：不调用任何模型，按楼层顺序做“拼接式摘要”。
    主要用途：无密钥/无网络时仍可产出符合 M2 schema 的可追溯 JSON，便于联调后续步骤。
    """
    paragraph_chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    for f in floors:
        line = f"{f.get('author','')}: {f.get('text','')}".strip()
        if not line:
            continue
        current.append(f)
        current_chars += len(line)
        if current_chars >= 900 or len(current) >= 8:
            paragraph_chunks.append(current)
            current = []
            current_chars = 0
    if current:
        paragraph_chunks.append(current)

    paragraphs: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    plot_points: List[str] = []

    for i, chunk in enumerate(paragraph_chunks, start=1):
        idxs = [int(x.get("index") or 0) for x in chunk if int(x.get("index") or 0) > 0]
        idxs = sorted(set(idxs))
        start_i = idxs[0] if idxs else meta["floor_min"]
        end_i = idxs[-1] if idxs else meta["floor_max"]
        text_lines: List[str] = []
        for f in chunk:
            t = str(f.get("text") or "").strip()
            if not t:
                continue
            text_lines.append(t)
        para_text = " ".join(text_lines).strip()
        paragraphs.append({"paragraph_id": i, "text": para_text})
        trace.append({"paragraph_id": i, "floors": {"start": start_i, "end": end_i, "indexes": idxs}})
        if para_text:
            plot_points.append(para_text[:80])

    synopsis = (paragraphs[0]["text"][:140] if paragraphs else "").strip()
    return {
        "title": meta.get("title") or "",
        "synopsis": synopsis,
        "plot_points": plot_points[:12],
        "paragraphs": paragraphs,
        "trace": trace,
        "uncertainties": [
            "当前为离线兜底输出：未调用 LLM/VLM，内容为楼层拼接摘要，仅用于联调；正式产出请关闭 --offline 并配置 API Key。"
        ],
    }


def summarize_m2(
    project_dir: str,
    llm: LlmConfig,
    vlm: Optional[VlmConfig],
    enable_vlm: bool,
    provider: str,
    prompt_pack: str,
    style_preset: str,
    persona: str,
    min_paragraph_seconds: float,
    target_min_seconds: float,
    target_max_seconds: float,
    chars_per_second: float,
    max_floors: int,
    max_chars: int,
    force: bool,
    retries: int,
    sleep_s: float,
) -> str:
    """
    M2：将 projects/{post_id}/source/raw.json 的楼层整理为 story.summary.json（含楼层映射）。
    """
    raw = _load_project_source(project_dir)
    meta, floors = _build_m2_input(raw=raw, max_floors=max_floors, max_chars=max_chars)
    input_obj = {"meta": meta, "floors": floors}
    input_hash = _sha256_json(input_obj)

    out_path = os.path.join(project_dir, "llm", "story.summary.json")
    if not force and os.path.exists(out_path):
        try:
            prev = json.loads(_read_text(out_path))
            if isinstance(prev, dict) and prev.get("input_hash") == input_hash:
                prev_meta = prev.get("meta") or {}
                prev_llm = prev_meta.get("llm")
                prev_vlm = prev_meta.get("vlm")
                prev_uncertainties = prev.get("uncertainties") or []
                is_offline_prev = isinstance(prev_uncertainties, list) and any("离线兜底输出" in str(x) for x in prev_uncertainties)
                if prev_llm is not None and not is_offline_prev:
                    if enable_vlm:
                        if isinstance(prev_vlm, dict) and str(prev_vlm.get("model") or "") == (vlm.model if vlm else ""):
                            return out_path
                    else:
                        return out_path
        except Exception:
            pass

    _maybe_backup_offline_summary(project_dir=project_dir, summary_path=out_path)

    image_cache = _load_image_desc_cache(project_dir)
    cache_items: Dict[str, Any] = image_cache.get("items") or {}
    if not isinstance(cache_items, dict):
        cache_items = {}

    floors_with_vlm: List[Dict[str, Any]] = []
    if enable_vlm and vlm is not None:
        for f in floors:
            imgs = f.get("images") or []
            if not isinstance(imgs, list):
                imgs = []
            descs: List[Dict[str, Any]] = []
            for u in imgs:
                uu = str(u).strip()
                if not uu:
                    continue
                cached = cache_items.get(uu)
                if isinstance(cached, dict) and cached.get("caption"):
                    descs.append(cached)
                    continue
                desc = _describe_image(vlm=vlm, image_url=uu, floor_index=int(f.get("index") or 0), prompt_pack=prompt_pack)
                cache_items[uu] = desc
                descs.append(desc)
                time.sleep(0.15)
            floors_with_vlm.append({**f, "image_descriptions": descs})
        image_cache["items"] = cache_items
        image_cache["meta"] = {"updated_at": datetime.now().isoformat(timespec="seconds"), "model": vlm.model}
        _save_image_desc_cache(project_dir, image_cache)
    else:
        floors_with_vlm = floors

    url = _chat_completions_url(llm.base_url) if provider != "claude" else ""
    tpl = get_template(prompt_pack, "summarize_v3")
    narration = _estimate_narration_constraints(
        min_paragraph_seconds=min_paragraph_seconds,
        target_min_seconds=target_min_seconds,
        target_max_seconds=target_max_seconds,
        chars_per_second=chars_per_second,
    )
    sys_prompt = (
        tpl.system_prompt
        + "\n\n"
        + "Timing constraints for voiceover paragraphs:\n"
        + f"- target_total_seconds: {int(narration['target_min_seconds'])}-{int(narration['target_max_seconds'])}\n"
        + f"- min_paragraph_seconds: {int(narration['min_paragraph_seconds'])}\n"
        + f"- chars_per_second: {narration['chars_per_second']:.1f}\n"
        + f"- paragraphs_count_range: {narration['min_paragraphs']}-{narration['max_paragraphs']}\n"
        + f"- paragraph_char_range: {narration['min_paragraph_chars']}-{narration['max_paragraph_chars']}\n"
        + f"- total_char_range: {narration['target_min_chars']}-{narration['target_max_chars']}\n"
    )
    user_prompt = render_user_prompt(
        tpl.user_prompt_template,
        {
            "post_url": meta.get("post_url") or "",
            "post_id": _extract_post_id_from_project_dir(project_dir),
            "style_preset": style_preset,
            "persona": persona,
            "floors_normalized_json": {"meta": meta, "floors": floors_with_vlm},
            "vlm_image_descriptions_json": image_cache.get("items") if (enable_vlm and vlm is not None) else [],
        },
    )

    suggested_max_tokens = min(llm.max_tokens, max(800, int(narration["target_max_chars"] * 2)))
    payload: Dict[str, Any] = {
        "model": llm.model,
        "temperature": llm.temperature,
        "max_tokens": suggested_max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    if provider == "claude":
        content = claude_print(user_prompt=user_prompt, system_prompt=sys_prompt)
    else:
        last_err: Optional[Exception] = None
        resp: Optional[Dict[str, Any]] = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                resp = _http_post_json(url=url, api_key=llm.api_key, payload=payload, timeout_s=llm.timeout_s)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(max(0.0, sleep_s))
        if last_err:
            raise last_err
        if not isinstance(resp, dict):
            raise RuntimeError("LLM 返回格式不正确：期望 JSON object")
        content = _extract_message_content(resp)
    last_parse_err: Optional[Exception] = None
    result_obj: Dict[str, Any] = {}
    for attempt in range(1, 4):
        try:
            result_obj = _extract_json_object(content)
            last_parse_err = None
            break
        except Exception as e:
            last_parse_err = e
            try:
                _ensure_dir(os.path.join(project_dir, "llm"))
                raw_path = os.path.join(project_dir, "llm", f"m2.last_raw.attempt{attempt}.txt")
                with open(raw_path, "w", encoding="utf-8") as fp:
                    fp.write(content)
            except Exception:
                pass
            repair_sys = (
                sys_prompt
                + "\n\n"
                + "你上一次输出无法被严格 JSON 解析。请重新输出严格 JSON，并满足以下额外限制：\n"
                + "- 必须输出单行 JSON（不要换行）\n"
                + "- 所有字符串值中不允许出现真实换行符，用空格代替\n"
                + "- 只能输出 1 个 JSON object；不得包含 markdown；不得包含解释文字；所有字符串必须用双引号；不得有尾随逗号\n"
            )
            repair_payload = {
                **payload,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": repair_sys},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if provider == "claude":
                content = claude_print(user_prompt=user_prompt, system_prompt=repair_sys)
            else:
                resp_fix = _http_post_json(url=url, api_key=llm.api_key, payload=repair_payload, timeout_s=llm.timeout_s)
                content = _extract_message_content(resp_fix)
            if attempt < 3:
                time.sleep(0.4)
    if last_parse_err is not None:
        raise last_parse_err
    title = str(result_obj.get("title") or meta.get("title") or "").strip()
    synopsis = str(result_obj.get("synopsis") or "").strip()
    plot_points = result_obj.get("plot_points") or []
    paragraphs = result_obj.get("paragraphs") or []
    uncertainties = result_obj.get("uncertainties") or []

    if not isinstance(plot_points, list):
        plot_points = []
    plot_points = [str(x).strip() for x in plot_points if str(x).strip()]

    if not isinstance(paragraphs, list):
        paragraphs = []

    for _ in range(4):
        paragraph_texts = [str((p or {}).get("text") or "").strip() for p in paragraphs if isinstance(p, dict)]
        ok, reasons = _check_narration_constraints(paragraph_texts=paragraph_texts, narration=narration)
        if ok:
            break
        repair_sys = (
            sys_prompt
            + "\n\n"
            + "你上一次输出未满足“强制要求”，请直接重写为满足要求的 JSON。\n"
            + "不允许输出除 JSON 以外的任何文本。\n"
            + "以下是不满足原因（供你修正）：\n"
            + "\n".join(f"- {r}" for r in reasons)
        )
        repair_payload = {
            **payload,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": repair_sys},
                {"role": "user", "content": user_prompt},
            ],
        }
        if provider == "claude":
            obj2 = _extract_json_object(claude_print(user_prompt=user_prompt, system_prompt=repair_sys))
        else:
            resp2 = _http_post_json(url=url, api_key=llm.api_key, payload=repair_payload, timeout_s=llm.timeout_s)
            content2 = _extract_message_content(resp2)
            obj2 = _extract_json_object(content2)
        paragraphs2 = obj2.get("paragraphs") or []
        if isinstance(paragraphs2, list):
            paragraphs = paragraphs2
            if isinstance(obj2.get("plot_points"), list):
                plot_points = obj2.get("plot_points") or plot_points
            synopsis = str(obj2.get("synopsis") or synopsis).strip()
            title = str(obj2.get("title") or title).strip()
            uncertainties = obj2.get("uncertainties") or uncertainties
    if isinstance(paragraphs, list):
        paragraphs = _enforce_narration_by_truncation(paragraphs=paragraphs, narration=narration)
    paragraph_texts = [str((p or {}).get("text") or "").strip() for p in paragraphs if isinstance(p, dict)]
    ok, reasons = _check_narration_constraints(paragraph_texts=paragraph_texts, narration=narration)
    if not ok:
        raise RuntimeError("故事段落未满足视频时长/字数约束（已尝试截断修正仍失败）：" + "; ".join(reasons))

    valid_floor_indexes = [f["index"] for f in floors if isinstance(f.get("index"), int)]
    fixed_paragraphs, trace = _coerce_trace(paragraphs=paragraphs, valid_floor_indexes=valid_floor_indexes)

    out_obj = {
        "post_url": meta["post_url"],
        "post_id": _extract_post_id_from_project_dir(project_dir),
        "title": title,
        "synopsis": synopsis,
        "plot_points": plot_points,
        "paragraphs": fixed_paragraphs,
        "trace": trace,
        "uncertainties": uncertainties if isinstance(uncertainties, list) else [],
        "input_hash": input_hash,
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_pack": prompt_pack_version(prompt_pack),
            "style_preset": style_preset,
            "persona": persona,
            "llm": {"base_url": llm.base_url, "model": llm.model, "temperature": llm.temperature, "max_tokens": llm.max_tokens},
            "vlm": {"base_url": vlm.base_url, "model": vlm.model} if (enable_vlm and vlm is not None) else None,
            "input": {"max_floors": max_floors, "max_chars": max_chars, "floor_min": meta["floor_min"], "floor_max": meta["floor_max"]},
            "narration": narration,
        },
    }
    _write_json(out_path, out_obj)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True, help="projects/{post_id} 目录（包含 source/raw.json）")
    p.add_argument("--offline", action="store_true", help="离线兜底：不调用模型，直接生成可追溯摘要（用于联调）")
    p.add_argument("--config", default=os.path.join(os.getcwd(), "docs", "aliyun_api-key.md"), help="本地密钥配置文件（默认 docs/aliyun_api-key.md）")
    p.add_argument("--base-url", default="", help="可选：覆盖配置文件里的 base_url（例如 https://dashscope.aliyuncs.com）")
    p.add_argument("--model", default="glm-4.7", help="用于 M2 整理的 LLM 模型名（OpenAI 兼容）")
    p.add_argument("--enable-vlm", action="store_true", help="启用 VLM：先对楼层图片做理解，再交给 LLM 总结")
    p.add_argument("--vlm-model", default="qwen3-vl-plus-2025-12-19", help="用于图片理解的 VLM 模型名（OpenAI 兼容）")
    p.add_argument("--provider", choices=["aliyun", "claude"], default="aliyun")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-floors", type=int, default=220)
    p.add_argument("--max-chars", type=int, default=60000)
    p.add_argument("--min-paragraph-seconds", type=float, default=20.0)
    p.add_argument("--target-min-seconds", type=float, default=120.0)
    p.add_argument("--target-max-seconds", type=float, default=180.0)
    p.add_argument("--chars-per-second", type=float, default=3.5)
    p.add_argument("--prompt-pack", default="v1")
    p.add_argument("--style-preset", default="documentary")
    p.add_argument("--persona", default="克制回望")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    try:
        project_dir = args.project_dir
        raw = _load_project_source(project_dir)
        meta, floors = _build_m2_input(raw=raw, max_floors=max(1, int(args.max_floors)), max_chars=max(1000, int(args.max_chars)))
        input_obj = {"meta": meta, "floors": floors}
        input_hash = _sha256_json(input_obj)

        out_path = os.path.join(project_dir, "llm", "story.summary.json")
        if not bool(args.force) and os.path.exists(out_path):
            try:
                prev = json.loads(_read_text(out_path))
                if isinstance(prev, dict) and prev.get("input_hash") == input_hash:
                    prev_meta = prev.get("meta") or {}
                    prev_llm = prev_meta.get("llm")
                    prev_vlm = prev_meta.get("vlm")
                    prev_uncertainties = prev.get("uncertainties") or []
                    is_offline_prev = isinstance(prev_uncertainties, list) and any("离线兜底输出" in str(x) for x in prev_uncertainties)
                    if args.offline:
                        print(json.dumps({"ok": True, "out": out_path, "cached": True, "offline": True}, ensure_ascii=False))
                        return 0
                    if prev_llm is not None and not is_offline_prev:
                        if args.enable_vlm:
                            if isinstance(prev_vlm, dict) and str(prev_vlm.get("model") or "") == str(args.vlm_model or ""):
                                print(json.dumps({"ok": True, "out": out_path, "cached": True}, ensure_ascii=False))
                                return 0
                        else:
                            print(json.dumps({"ok": True, "out": out_path, "cached": True}, ensure_ascii=False))
                            return 0
            except Exception:
                pass

        if args.offline:
            offline_obj = _summarize_offline(meta=meta, floors=floors)
            out_obj = {
                "post_url": meta["post_url"],
                "post_id": _extract_post_id_from_project_dir(project_dir),
                "title": offline_obj["title"],
                "synopsis": offline_obj["synopsis"],
                "plot_points": offline_obj["plot_points"],
                "paragraphs": offline_obj["paragraphs"],
                "trace": offline_obj["trace"],
                "uncertainties": offline_obj["uncertainties"],
                "input_hash": input_hash,
                "meta": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "llm": None,
                    "vlm": None,
                    "input": {"max_floors": int(args.max_floors), "max_chars": int(args.max_chars), "floor_min": meta["floor_min"], "floor_max": meta["floor_max"]},
                },
            }
            _write_json(out_path, out_obj)
            print(json.dumps({"ok": True, "out": out_path, "offline": True}, ensure_ascii=False))
            return 0

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
            vlm = None
            enable_vlm = False
        else:
            llm = _load_llm_config(
                config_path=args.config,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                base_url=(args.base_url or "").strip() or None,
            )
            vlm = None
            enable_vlm = bool(args.enable_vlm)
            if enable_vlm:
                vlm = _load_vlm_config(
                    config_path=args.config,
                    model=args.vlm_model,
                    timeout_s=args.timeout,
                    base_url=(args.base_url or "").strip() or None,
                )

        out_path = summarize_m2(
            project_dir=project_dir,
            llm=llm,
            vlm=vlm,
            enable_vlm=enable_vlm,
            provider=provider,
            prompt_pack=str(args.prompt_pack),
            style_preset=str(args.style_preset),
            persona=str(args.persona),
            min_paragraph_seconds=float(args.min_paragraph_seconds),
            target_min_seconds=float(args.target_min_seconds),
            target_max_seconds=float(args.target_max_seconds),
            chars_per_second=float(args.chars_per_second),
            max_floors=max(1, int(args.max_floors)),
            max_chars=max(1000, int(args.max_chars)),
            force=bool(args.force),
            retries=max(1, int(args.retries)),
            sleep_s=max(0.0, float(args.sleep)),
        )
        print(json.dumps({"ok": True, "out": out_path}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{e.__class__.__name__}: {e}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
