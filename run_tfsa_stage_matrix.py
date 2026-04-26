#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


MODEL_PATH = Path("/data/wyx/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d")
SRC = Path("/data/wyx/Projects/MorphAny3D/assets/example_morphing/Godzilla.png")
TAR = Path("/data/wyx/Projects/MorphAny3D/assets/example_morphing/typical_humanoid_dragonborn.png")
COMPARE_ROOT = Path("/data/wyx/Projects/Morph比较")
CACHE_ROOT = COMPARE_ROOT / "cache"
GALLERY_ROOT = COMPARE_ROOT / "MorphAny3D2_512_tfsa_stage_matrix_照片汇总"
PROJECT_ROOT = Path("/data/wyx/Projects/MorphAny3D2")


def count_frames(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return len(list(output_dir.glob("frame_*.png")))


def link_pngs(name: str, output_dir: Path) -> None:
    target_dir = GALLERY_ROOT / name
    target_dir.mkdir(parents=True, exist_ok=True)
    for link in target_dir.glob("frame_*.png"):
        if link.is_symlink() or link.exists():
            link.unlink()
    for png in sorted(output_dir.glob("frame_*.png")):
        (target_dir / png.name).symlink_to(png)


def run_one(name: str, gpu: str, ss: bool, shape: bool, tex: bool, symmetric_tar_cfg: bool) -> None:
    output_dir = COMPARE_ROOT / name
    cache_dir = CACHE_ROOT / name
    tfsa_cache_dir = Path("/tmp") / name
    log_path = COMPARE_ROOT / f"{name}.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tfsa_cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "morph_frames_png_glb_with_video.py"),
        "--gpu",
        gpu,
        "--model-path",
        str(MODEL_PATH),
        "--src",
        str(SRC),
        "--tar",
        str(TAR),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--tfsa-cache-dir",
        str(tfsa_cache_dir),
        "--morphing-num",
        "50",
        "--pipeline-type",
        "512",
        "--enable-tfsa",
        "--video-resolution",
        "256",
        "--video-num-frames",
        "60",
    ]
    if symmetric_tar_cfg:
        cmd.append("--symmetric-tar-cfg")
    if not ss:
        cmd.append("--disable-ss-tfsa")
    if not shape:
        cmd.append("--disable-shape-tfsa")
    if not tex:
        cmd.append("--disable-tex-tfsa")

    env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        env.pop(key, None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    with log_path.open("w") as log_file:
        log_file.write(f"Running {name}: ss={ss}, shape={shape}, tex={tex}\n")
        log_file.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    link_pngs(name, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", nargs="?", default="2")
    parser.add_argument("--symmetric-tar-cfg", action="store_true")
    parser.add_argument("--all-eight", action="store_true")
    args = parser.parse_args()

    gpu = args.gpu
    GALLERY_ROOT.mkdir(parents=True, exist_ok=True)

    existing = {
        "MorphAny3D2_512_tfsa_nosymcfg_default": COMPARE_ROOT / "MorphAny3D2_512_tfsa_nosymcfg_default",
        "MorphAny3D2_512_no_tex_tfsa": COMPARE_ROOT / "MorphAny3D2_512_no_tex_tfsa",
    }
    for name, path in existing.items():
        if path.exists():
            link_pngs(name, path)

    prefix = "MorphAny3D2_512_symcfg_" if args.symmetric_tar_cfg else "MorphAny3D2_512_"
    if args.all_eight:
        combos = [
            (f"{prefix}no_ss_no_shape_no_tex_tfsa", False, False, False),
            (f"{prefix}ss_only_tfsa", True, False, False),
            (f"{prefix}shape_only_tfsa", False, True, False),
            (f"{prefix}tex_only_tfsa", False, False, True),
            (f"{prefix}ss_shape_tfsa", True, True, False),
            (f"{prefix}ss_tex_tfsa", True, False, True),
            (f"{prefix}shape_tex_tfsa", False, True, True),
            (f"{prefix}all_tfsa", True, True, True),
        ]
    else:
        combos = [
            (f"{prefix}no_ss_no_shape_no_tex_tfsa", False, False, False),
            (f"{prefix}ss_only_tfsa", True, False, False),
            (f"{prefix}shape_only_tfsa", False, True, False),
            (f"{prefix}tex_only_tfsa", False, False, True),
            (f"{prefix}ss_tex_tfsa", True, False, True),
            (f"{prefix}shape_tex_tfsa", False, True, True),
        ]

    for name, ss, shape, tex in combos:
        output_dir = COMPARE_ROOT / name
        if count_frames(output_dir) >= 50:
            link_pngs(name, output_dir)
            continue
        run_one(name, gpu, ss, shape, tex, args.symmetric_tar_cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
