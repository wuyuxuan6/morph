import os
import argparse
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import torch

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import seed_everything, cleanup_cuda, get_default_envmap
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


def export_preview(mesh, output_path: Path, envmap, resolution: int = 512) -> None:
    snapshot = render_utils.render_snapshot(mesh, resolution=resolution, nviews=4, envmap=envmap)
    if "shaded" in snapshot:
        frames = render_utils.make_pbr_vis_frames(snapshot, resolution=resolution)
        Image.fromarray(frames[0]).save(output_path)
    elif "color" in snapshot:
        Image.fromarray(snapshot["color"][0]).save(output_path)
    else:
        first_key = next(iter(snapshot))
        Image.fromarray(snapshot[first_key][0]).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MorphAny3D2 and export per-frame PNG + GLB.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morphing-num", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=500000)
    parser.add_argument("--preview-resolution", type=int, default=512)
    parser.add_argument("--enable-tfsa", action="store_true")
    parser.add_argument("--disable-oc", action="store_true")
    parser.add_argument("--init-morphing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    envmap = get_default_envmap(device=f"cuda:{args.gpu}")

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    src_name = Path(args.src).stem
    tar_name = Path(args.tar).stem

    cache_root = output_dir / "cache"
    src_cache_dir = cache_root / src_name / "cache"
    tar_cache_dir = cache_root / tar_name / "cache"
    morph_cache_dir = cache_root / "morph"
    src_cache_dir.mkdir(parents=True, exist_ok=True)
    tar_cache_dir.mkdir(parents=True, exist_ok=True)
    morph_cache_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    def save_frame(mesh, frame_idx: int) -> None:
        glb_path = output_dir / f"frame_{frame_idx:03d}.glb"
        png_path = output_dir / f"frame_{frame_idx:03d}.png"
        export_glb(mesh, glb_path, texture_size=args.texture_size, decimation_target=args.decimation_target)
        export_preview(mesh, png_path, envmap=envmap, resolution=args.preview_resolution)

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

    if not (src_cache_dir / "tex_slat_init.pt").exists():
        src_outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params={**cache_params, "save_cache_path": str(src_cache_dir)},
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )
        save_frame(src_outputs[0], 0)
        del src_outputs
        cleanup_cuda()

    if not (tar_cache_dir / "tex_slat_init.pt").exists():
        tar_outputs = pipeline.run_morphing(
            src_img=tar_img,
            tar_img=src_img,
            morphing_params={**cache_params, "save_cache_path": str(tar_cache_dir)},
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )
        save_frame(tar_outputs[0], args.morphing_num - 1)
        del tar_outputs
        cleanup_cuda()

    morphing_params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": str(src_cache_dir),
        "tar_load_cache_path": str(tar_cache_dir),
        "save_cache_path": str(morph_cache_dir),
        "init_morphing_flag": args.init_morphing,
        "ss_mca_flag": True,
        "ss_tfsa_flag": args.enable_tfsa,
        "shape_mca_flag": True,
        "shape_tfsa_flag": args.enable_tfsa,
        "tex_mca_flag": True,
        "tex_tfsa_flag": args.enable_tfsa,
        "oc_flag": not args.disable_oc,
        "tfsa_alpha": 0.8,
        "pipeline_type": args.pipeline_type,
    }

    alpha_array = torch.linspace(1.0, 0.0, args.morphing_num).tolist()
    for morphing_idx in range(1, args.morphing_num - 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )
        save_frame(outputs[0], morphing_idx)
        del outputs
        cleanup_cuda()

        old_files = list(morph_cache_dir.glob(f"ss_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(morph_cache_dir.glob(f"shape_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(morph_cache_dir.glob(f"tex_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        for file_path in old_files:
            file_path.unlink(missing_ok=True)

        print(f"[{morphing_idx + 1}/{args.morphing_num}] Saved frame_{morphing_idx:03d}", flush=True)


if __name__ == "__main__":
    main()
