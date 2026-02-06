import argparse
import os
import shutil


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_file(src: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def _copy_dir(src_dir: str, dst_dir: str, pattern_prefix: str) -> int:
    if not os.path.isdir(src_dir):
        return 0
    _ensure_dir(dst_dir)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        if not name.startswith(pattern_prefix):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dst_dir, name)
        _copy_file(src, dst)
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", required=True)
    p.add_argument("--remotion-dir", default="remotion_video")
    p.add_argument("--project-id", default=None)
    args = p.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    remotion_dir = os.path.abspath(args.remotion_dir)

    project_id = args.project_id
    if not project_id:
        project_id = os.path.basename(project_dir.rstrip("\\/"))

    src_images = os.path.join(project_dir, "render", "images")
    src_audio = os.path.join(project_dir, "render", "audio")
    src_bgm = os.path.join(project_dir, "assets", "bgm", "bgm.mp3")

    dst_root = os.path.join(remotion_dir, "public", str(project_id))
    dst_images = os.path.join(dst_root, "images")
    dst_audio = os.path.join(dst_root, "audio")
    dst_bgm_dir = os.path.join(dst_root, "bgm")
    dst_bgm = os.path.join(dst_bgm_dir, "bgm.mp3")

    img_n = _copy_dir(src_images, dst_images, "scene_")
    aud_n = _copy_dir(src_audio, dst_audio, "scene_")
    if os.path.exists(src_bgm):
        _copy_file(src_bgm, dst_bgm)
        bgm_ok = True
    else:
        bgm_ok = False

    print(
        {
            "ok": True,
            "project_id": str(project_id),
            "images_copied": img_n,
            "audio_copied": aud_n,
            "bgm_copied": bgm_ok,
            "dst_root": dst_root,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

