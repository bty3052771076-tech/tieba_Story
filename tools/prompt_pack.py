import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    purpose: str
    system_prompt: str
    user_prompt_template: str


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_prompt_pack(pack: str) -> Dict[str, Any]:
    root = _repo_root()
    path = os.path.join(root, "prompts", "packs", pack, "prompt_pack.json")
    if not os.path.exists(path):
        raise RuntimeError(f"未找到 prompt pack：{path}")
    obj = _read_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("templates"), dict):
        raise RuntimeError(f"prompt pack 格式不正确：{path}")
    return obj


def get_template(pack: str, name: str) -> PromptTemplate:
    obj = load_prompt_pack(pack)
    templates = obj.get("templates") or {}
    t = templates.get(name)
    if not isinstance(t, dict):
        raise RuntimeError(f"prompt pack 缺少模板：{name}")
    system_prompt = str(t.get("system_prompt") or "").strip()
    user_prompt_template = str(t.get("user_prompt_template") or "").strip()
    purpose = str(t.get("purpose") or "").strip()
    if not system_prompt or not user_prompt_template:
        raise RuntimeError(f"模板内容为空：{name}")
    return PromptTemplate(name=name, purpose=purpose, system_prompt=system_prompt, user_prompt_template=user_prompt_template)


def render_user_prompt(user_prompt_template: str, variables: Dict[str, Any]) -> str:
    safe_vars: Dict[str, str] = {}
    for k, v in variables.items():
        if v is None:
            safe_vars[k] = ""
        elif isinstance(v, str):
            safe_vars[k] = v
        else:
            safe_vars[k] = json.dumps(v, ensure_ascii=False, indent=2)
    try:
        return user_prompt_template.format(**safe_vars)
    except KeyError as e:
        raise RuntimeError(f"提示词模板缺少变量：{e}")


def prompt_pack_version(pack: str) -> str:
    obj = load_prompt_pack(pack)
    v = str(obj.get("version") or "").strip()
    return v or pack

