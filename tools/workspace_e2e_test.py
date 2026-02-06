import argparse
import glob
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


def _now_s() -> float:
    return time.time()


def _run(cmd: List[str], cwd: Optional[str], timeout_s: int) -> Dict[str, Any]:
    started = _now_s()
    if os.name == "nt" and cmd:
        if cmd[0] in {"npx", "npm", "node"}:
            cmd = [cmd[0] + ".cmd"] + cmd[1:]
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            shell=False,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "elapsed_s": round(_now_s() - started, 3),
            "cmd": cmd,
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": (e.stdout or ""),
            "stderr": (e.stderr or ""),
            "elapsed_s": round(_now_s() - started, 3),
            "cmd": cmd,
            "cwd": cwd,
            "timeout": True,
        }


def _file_info(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"exists": False}
    st = os.stat(path)
    return {"exists": True, "bytes": int(st.st_size)}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def _test_python_help(workspace: str, timeout_s: int) -> Tuple[int, int, List[Dict[str, Any]]]:
    tools_dir = os.path.join(workspace, "tools")
    scripts = sorted(glob.glob(os.path.join(tools_dir, "*.py")))
    results = []
    pass_count = 0
    fail_count = 0
    for s in scripts:
        r = _run(["python", s, "--help"], cwd=workspace, timeout_s=timeout_s)
        item = {
            "script": os.path.relpath(s, workspace).replace("\\", "/"),
            "ok": bool(r["ok"]),
            "returncode": r["returncode"],
            "elapsed_s": r["elapsed_s"],
        }
        if not r["ok"]:
            item["stderr_tail"] = (r["stderr"] or "")[-400:]
        results.append(item)
        if r["ok"]:
            pass_count += 1
        else:
            fail_count += 1
    return pass_count, fail_count, results


def _validate_props(props_path: str) -> Dict[str, Any]:
    info = _file_info(props_path)
    if not info.get("exists"):
        return {"ok": False, "reason": "props 不存在", "path": props_path}
    try:
        props = _read_json(props_path)
    except Exception as e:
        return {"ok": False, "reason": "props 解析失败", "error": str(e), "path": props_path}
    scenes = props.get("scenes") if isinstance(props, dict) else None
    if not isinstance(scenes, list) or not scenes:
        return {"ok": False, "reason": "props.scenes 为空", "path": props_path}
    first3 = scenes[:3]
    missing = []
    for i, s in enumerate(first3):
        if not isinstance(s, dict):
            missing.append({"index": i, "reason": "scene 不是对象"})
            continue
        for k in ["image", "audio", "duration_s", "subtitle"]:
            v = s.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                missing.append({"index": i, "field": k})
    return {"ok": len(missing) == 0, "missing": missing, "scene_count": len(scenes), "path": props_path}


def _render_smoke(remotion_dir: str, props_path: str, out_path: str) -> Dict[str, Any]:
    cmd_base = ["npx", "remotion", "render", "TiebaStory", out_path, "--props", props_path]
    attempt1 = _run(cmd_base + ["--frames", "0-30"], cwd=remotion_dir, timeout_s=180)
    if attempt1["ok"] and _file_info(out_path).get("bytes", 0) > 0:
        return {"ok": True, "method": "frames_0_30", "run": attempt1, "out": _file_info(out_path)}
    attempt2 = _run(cmd_base, cwd=remotion_dir, timeout_s=300)
    return {"ok": bool(attempt2["ok"]) and _file_info(out_path).get("bytes", 0) > 0, "method": "full", "run": attempt2, "out": _file_info(out_path)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default=os.getcwd())
    p.add_argument("--project-id", default="9461449190")
    args = p.parse_args(argv)

    started = _now_s()
    workspace = os.path.abspath(args.workspace)
    project_dir = os.path.join(workspace, "projects", str(args.project_id))
    remotion_dir = os.path.join(workspace, "remotion_video")

    report: Dict[str, Any] = {"workspace": workspace, "project_id": str(args.project_id)}

    py_pass, py_fail, py_help = _test_python_help(workspace, timeout_s=15)
    report["python_help"] = {"pass": py_pass, "fail": py_fail, "items": py_help}

    m5 = _run(["python", os.path.join(workspace, "tools", "tieba_m5_validate.py"), "--project-dir", project_dir], cwd=workspace, timeout_s=30)
    report["python_m5_validate"] = {"ok": bool(m5["ok"]), "returncode": m5["returncode"], "elapsed_s": m5["elapsed_s"], "stderr_tail": (m5["stderr"] or "")[-400:]}

    props_out = os.path.join(remotion_dir, f"props.{args.project_id}.e2e.json")
    m6 = _run(
        [
            "python",
            os.path.join(workspace, "tools", "tieba_m6_build_props.py"),
            "--project-dir",
            project_dir,
            "--public-prefix",
            str(args.project_id),
            "--bgm-rel",
            f"{args.project_id}/bgm/bgm.mp3",
            "--max-scenes",
            "3",
            "--out",
            props_out,
        ],
        cwd=workspace,
        timeout_s=30,
    )
    report["python_m6_build_props"] = {"ok": bool(m6["ok"]), "returncode": m6["returncode"], "elapsed_s": m6["elapsed_s"], "out": _file_info(props_out), "stderr_tail": (m6["stderr"] or "")[-400:]}

    comp = _run(["npx", "remotion", "compositions"], cwd=remotion_dir, timeout_s=120)
    report["remotion_compositions"] = {"ok": bool(comp["ok"]), "returncode": comp["returncode"], "elapsed_s": comp["elapsed_s"], "has_TiebaStory": "TiebaStory" in (comp["stdout"] or ""), "stderr_tail": (comp["stderr"] or "")[-400:]}

    props_for_video = os.path.join(remotion_dir, f"props.{args.project_id}.m6.v2.json")
    if not os.path.exists(props_for_video):
        props_for_video = os.path.join(workspace, "remotion_video", f"props.{args.project_id}.m6.v2.json")
    out_test_video = os.path.join(workspace, "output", f"{args.project_id}.m6.test.mp4")
    report["remotion_render_smoke"] = _render_smoke(remotion_dir, props_for_video, out_test_video)

    out_existing = os.path.join(workspace, "output", f"{args.project_id}.m6.v2.mp4")
    report["outputs"] = {
        "existing_video": {"path": out_existing, "info": _file_info(out_existing)},
        "test_video": {"path": out_test_video, "info": _file_info(out_test_video)},
        "props_check": _validate_props(props_for_video),
    }

    elapsed_s = round(_now_s() - started, 3)
    fail_count = 0
    fail_count += 1 if report["python_help"]["fail"] > 0 else 0
    fail_count += 1 if not report["python_m5_validate"]["ok"] else 0
    fail_count += 1 if not report["python_m6_build_props"]["ok"] else 0
    fail_count += 1 if not report["remotion_compositions"]["ok"] or not report["remotion_compositions"]["has_TiebaStory"] else 0
    fail_count += 1 if not report["remotion_render_smoke"]["ok"] else 0
    fail_count += 1 if not report["outputs"]["existing_video"]["info"].get("exists") or report["outputs"]["existing_video"]["info"].get("bytes", 0) <= 0 else 0
    fail_count += 1 if not report["outputs"]["test_video"]["info"].get("exists") or report["outputs"]["test_video"]["info"].get("bytes", 0) <= 0 else 0
    fail_count += 1 if not report["outputs"]["props_check"]["ok"] else 0
    report["summary"] = {"fail_count": fail_count, "elapsed_s": elapsed_s}

    report_path = os.path.join(workspace, "output", f"workspace_test_report.{args.project_id}.json")
    _write_json(report_path, report)
    print(json.dumps({"ok": fail_count == 0, "report": report_path, "fail_count": fail_count, "elapsed_s": elapsed_s}, ensure_ascii=False))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
