import os
import argparse
from pathlib import Path
import tempfile
import shutil
import fcntl

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import (
    seed_everything,
    cleanup_cuda,
    get_default_envmap,
    ROT_VIDEO_RESOLUTION,
    ROT_VIDEO_NUM_FRAMES,
)
from trellis2.utils import render_utils


def export_glb(mesh, output_path: Path, texture_size: int = 2048, decimation_target: int = 500000) -> None:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(output_path), extension_webp=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MorphAny3D2 and export per-frame PNG + GLB plus original-style videos.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--pipeline-type", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morphing-num", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=500000)
    parser.add_argument("--preview-resolution", type=int, default=512)
    parser.add_argument("--video-resolution", type=int, default=ROT_VIDEO_RESOLUTION)
    parser.add_argument("--video-num-frames", type=int, default=ROT_VIDEO_NUM_FRAMES)
    parser.add_argument("--enable-tfsa", action="store_true")
    parser.add_argument("--symmetric-tar-cfg", action="store_true")
    parser.add_argument("--legacy-cfg-formula", action="store_true")
    parser.add_argument("--legacy-tfsa-cache-alignment", action="store_true")
    parser.add_argument("--disable-ss-tfsa", action="store_true")
    parser.add_argument("--disable-shape-tfsa", action="store_true")
    parser.add_argument("--disable-tex-tfsa", action="store_true")
    parser.add_argument("--disable-shape-lr-tfsa", action="store_true")
    parser.add_argument("--disable-shape-hr-tfsa", action="store_true")
    parser.add_argument("--disable-oc", action="store_true")
    parser.add_argument("--init-morphing", action="store_true")
    parser.add_argument("--endpoints-only", action="store_true")
    parser.add_argument("--skip-glb", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tfsa-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cleanup_cache_root = False
    else:
        temp_base = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
        cache_root = Path(tempfile.mkdtemp(prefix=f"morphany3d2_cache_{output_dir.name}_", dir=str(temp_base)))
        cleanup_cache_root = True

    if args.tfsa_cache_dir:
        tfsa_cache_root = Path(args.tfsa_cache_dir)
        tfsa_cache_root.mkdir(parents=True, exist_ok=True)
        cleanup_tfsa_cache_root = False
    else:
        tfsa_base = Path("/dev/shm") if Path("/dev/shm").exists() else cache_root
        tfsa_cache_root = Path(tempfile.mkdtemp(prefix=f"morphany3d2_tfsa_{output_dir.name}_", dir=str(tfsa_base)))
        cleanup_tfsa_cache_root = True

    lock_path = Path(tempfile.gettempdir()) / "morphany3d2_trellis2_model_load.lock"
    print(f"Loading pipeline from {args.model_path}", flush=True)
    with open(lock_path, "w") as lock_file:
        print(f"Waiting for model load lock: {lock_path}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        print("Acquired model load lock", flush=True)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    envmap = get_default_envmap(device=f"cuda:{args.gpu}")
    print("Pipeline loaded and moved to CUDA", flush=True)

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    src_name = Path(args.src).stem
    tar_name = Path(args.tar).stem
    pair_name = f"{src_name}+{tar_name}"

    src_cache_dir = cache_root / src_name / "cache"
    tar_cache_dir = cache_root / tar_name / "cache"
    morph_cache_dir = cache_root / "morph"
    src_cache_dir.mkdir(parents=True, exist_ok=True)
    tar_cache_dir.mkdir(parents=True, exist_ok=True)
    morph_cache_dir.mkdir(parents=True, exist_ok=True)
    tfsa_cache_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    def save_preview(mesh, output_path: Path) -> None:
        snapshot = render_utils.render_snapshot(mesh, resolution=args.preview_resolution, nviews=4, envmap=envmap)
        if "shaded" in snapshot:
            frames = render_utils.make_pbr_vis_frames(snapshot, resolution=args.preview_resolution)
            Image.fromarray(frames[0]).save(output_path)
        elif "color" in snapshot:
            Image.fromarray(snapshot["color"][0]).save(output_path)
        else:
            first_key = next(iter(snapshot))
            Image.fromarray(snapshot[first_key][0]).save(output_path)

    def save_frame(mesh, frame_idx: int) -> None:
        print(f"frame_{frame_idx:03d}: saving preview png", flush=True)
        save_preview(mesh, output_dir / f"frame_{frame_idx:03d}.png")
        if not args.skip_glb:
            print(f"frame_{frame_idx:03d}: saving glb", flush=True)
            export_glb(mesh, output_dir / f"frame_{frame_idx:03d}.glb", texture_size=args.texture_size, decimation_target=args.decimation_target)

    def save_endpoint_video(mesh, prefix: str) -> None:
        print(f"{prefix}: rendering endpoint video", flush=True)
        video = render_utils.render_rot_video(
            mesh,
            resolution=args.video_resolution,
            num_frames=args.video_num_frames,
            envmap=envmap,
        )
        frames = render_utils.make_pbr_vis_frames(video, resolution=min(args.video_resolution, 1024))
        imageio.mimsave(output_dir / f"{prefix}.mp4", frames, fps=15)

    morphing_params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": str(src_cache_dir),
        "tar_load_cache_path": str(tar_cache_dir),
        "save_cache_path": str(morph_cache_dir),
        "init_morphing_flag": args.init_morphing,
        "ss_mca_flag": True,
        "ss_tfsa_flag": args.enable_tfsa and not args.disable_ss_tfsa,
        "shape_mca_flag": True,
        "shape_tfsa_flag": args.enable_tfsa and not args.disable_shape_tfsa,
        "shape_lr_tfsa_flag": args.enable_tfsa and not args.disable_shape_lr_tfsa,
        "shape_hr_tfsa_flag": args.enable_tfsa and not args.disable_shape_hr_tfsa,
        "tex_mca_flag": True,
        "tex_tfsa_flag": args.enable_tfsa and not args.disable_tex_tfsa,
        "oc_flag": not args.disable_oc,
        "tfsa_alpha": 0.8,
        "pipeline_type": args.pipeline_type,
        "cfg_symmetric_tar_cond": args.symmetric_tar_cfg,
        "legacy_cfg_formula": args.legacy_cfg_formula,
        "tfsa_cache_path": str(tfsa_cache_root),
        "legacy_tfsa_cache_alignment": args.legacy_tfsa_cache_alignment,
    }

    cache_params = {
        "pipeline_type": args.pipeline_type,
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "ss_tfsa_flag": False,
        "shape_mca_flag": False,
        "shape_tfsa_flag": False,
        "tex_mca_flag": False,
        "tex_tfsa_flag": False,
    }

    print(f"Using cache root: {cache_root}", flush=True)
    print(f"Using TFSA cache root: {tfsa_cache_root}", flush=True)

    if args.endpoints_only:
        endpoint_settings = [
            (0, 1.0, -1, src_img, tar_img, src_name),
            (args.morphing_num - 1, 0.0, 0, src_img, tar_img, tar_name),
        ]
        for frame_idx, alpha, tfsa_cache_idx, cur_src, cur_tar, prefix in endpoint_settings:
            endpoint_params = {
                **morphing_params,
                "alpha": alpha,
                "morphing_idx": frame_idx,
                "tfsa_cache_idx": tfsa_cache_idx,
            }
            print(f"Endpoint-only frame_{frame_idx:03d}: run_morphing start (alpha={alpha:.4f})", flush=True)
            outputs = pipeline.run_morphing(
                src_img=cur_src,
                tar_img=cur_tar,
                morphing_params=endpoint_params,
                seed=args.seed,
                pipeline_type=args.pipeline_type,
            )
            print(f"Endpoint-only frame_{frame_idx:03d}: run_morphing done", flush=True)
            save_frame(outputs[0], frame_idx)
            save_endpoint_video(outputs[0], prefix)
            del outputs
            cleanup_cuda()
        return

    print("Building source cache...", flush=True)
    src_outputs = pipeline.run_morphing(
        src_img=src_img,
        tar_img=tar_img,
        morphing_params={**cache_params, "save_cache_path": str(src_cache_dir)},
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )
    save_frame(src_outputs[0], 0)
    save_endpoint_video(src_outputs[0], src_name)
    del src_outputs
    cleanup_cuda()

    print("Building target cache...", flush=True)
    tar_outputs = pipeline.run_morphing(
        src_img=tar_img,
        tar_img=src_img,
        morphing_params={**cache_params, "save_cache_path": str(tar_cache_dir)},
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )
    save_frame(tar_outputs[0], args.morphing_num - 1)
    save_endpoint_video(tar_outputs[0], tar_name)
    del tar_outputs
    cleanup_cuda()

    alpha_array = np.linspace(1.0, 0.0, args.morphing_num)
    frames = []
    single_view_frames = []
    for morphing_idx in range(1, args.morphing_num - 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        print(f"[{morphing_idx + 1}/{args.morphing_num}] alpha={morphing_params['alpha']:.4f}", flush=True)
        print(f"frame_{morphing_idx:03d}: run_morphing start", flush=True)
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )
        print(f"frame_{morphing_idx:03d}: run_morphing done", flush=True)
        mesh = outputs[0]
        save_frame(mesh, morphing_idx)
        print(f"frame_{morphing_idx:03d}: rendering rot video", flush=True)
        video = render_utils.render_rot_video(
            mesh,
            resolution=args.video_resolution,
            num_frames=args.video_num_frames,
            envmap=envmap,
        )
        vis_frames = np.stack(render_utils.make_pbr_vis_frames(video, resolution=min(args.video_resolution, 1024)), axis=0)
        frames.append(vis_frames)
        single_view_frames.append(video["shaded"][0])
        del outputs
        cleanup_cuda()

        old_files = list(tfsa_cache_root.glob(f"ss_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(tfsa_cache_root.glob(f"shape_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(tfsa_cache_root.glob(f"tex_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(tfsa_cache_root.glob(f"ss_feat_coords_morphing{morphing_params['tfsa_cache_idx']}.pt"))
        old_files += list(tfsa_cache_root.glob(f"shape_feat_coords_morphing{morphing_params['tfsa_cache_idx']}.pt"))
        old_files += list(tfsa_cache_root.glob(f"tex_feat_coords_morphing{morphing_params['tfsa_cache_idx']}.pt"))
        for file_path in old_files:
            file_path.unlink(missing_ok=True)

    suffix = "initmorph" if args.init_morphing else "noinit"
    if frames:
        montage = np.concatenate(frames, axis=2)
        imageio.mimsave(output_dir / f"morphing_{pair_name}_{suffix}.mp4", montage, fps=max(1, (args.morphing_num - 2) // 2))
    if single_view_frames:
        imageio.mimsave(
            output_dir / f"morphing_{pair_name}_{suffix}_singleview.mp4",
            np.stack(single_view_frames, axis=0),
            fps=max(1, (args.morphing_num - 2) // 2),
        )

    if cleanup_cache_root:
        shutil.rmtree(cache_root, ignore_errors=True)
    if cleanup_tfsa_cache_root:
        shutil.rmtree(tfsa_cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
