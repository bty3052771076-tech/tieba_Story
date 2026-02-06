import json
import subprocess
from typing import Any, Dict, Optional


def claude_print(user_prompt: str, system_prompt: Optional[str] = None, tools: str = '""') -> str:
    cmd = ["claude", "-p", "--output-format", "json", "--tools", tools]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    p = subprocess.run(
        cmd,
        input=user_prompt,
        text=True,
        capture_output=True,
        shell=False,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip() or f"claude exited {p.returncode}")
    raw = (p.stdout or "").strip()
    if not raw:
        raise RuntimeError("claude 输出为空")
    obj: Dict[str, Any] = json.loads(raw)
    result = obj.get("result")
    if not isinstance(result, str):
        raise RuntimeError("claude 输出缺少 result 字段")
    return result
