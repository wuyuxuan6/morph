import os
import argparse
from pathlib import Path
import tempfile
import shutil
import fcntl

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import torch
import trimesh

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import seed_everything, cleanup_cuda, get_default_envmap
from trellis2.utils import render_utils


def export_mesh_glb(mesh, output_path: Path) -> None:
    tri = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    tri.export(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MorphAny3D2 1536_cascade morph and export only shape_lr PNG + GLB.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morphing-num", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--preview-resolution", type=int, default=512)
    parser.add_argument("--init-morphing", action="store_true")
    parser.add_argument("--cache-dir", default=None)
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
        cache_root = Path(tempfile.mkdtemp(prefix=f"morphany3d2_shape_lr_{output_dir.name}_", dir=str(temp_base)))
        cleanup_cache_root = True

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

    src_cache_dir = cache_root / src_name / "cache"
    tar_cache_dir = cache_root / tar_name / "cache"
    morph_cache_dir = cache_root / "morph"
    src_cache_dir.mkdir(parents=True, exist_ok=True)
    tar_cache_dir.mkdir(parents=True, exist_ok=True)
    morph_cache_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    def save_preview(mesh, output_path: Path) -> None:
        snapshot = render_utils.render_snapshot(mesh, resolution=args.preview_resolution, nviews=4)
        if "shaded" in snapshot:
            frames = render_utils.make_pbr_vis_frames(snapshot, resolution=args.preview_resolution)
            Image.fromarray(frames[0]).save(output_path)
        elif "normal" in snapshot:
            Image.fromarray(snapshot["normal"][0]).save(output_path)
        else:
            first_key = next(iter(snapshot))
            Image.fromarray(snapshot[first_key][0]).save(output_path)

    def save_frame(mesh, frame_idx: int) -> None:
        print(f"frame_{frame_idx:03d}: saving shape_lr preview png", flush=True)
        save_preview(mesh, output_dir / f"frame_{frame_idx:03d}.png")
        print(f"frame_{frame_idx:03d}: saving shape_lr glb", flush=True)
        export_mesh_glb(mesh, output_dir / f"frame_{frame_idx:03d}.glb")

    cache_params = {
        "pipeline_type": "1536_cascade",
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "ss_tfsa_flag": False,
        "shape_mca_flag": False,
        "shape_tfsa_flag": False,
        "tex_mca_flag": False,
        "tex_tfsa_flag": False,
        "return_shape_lr": True,
    }

    print(f"Using cache root: {cache_root}", flush=True)
    print("Building source cache...", flush=True)
    _, latent = pipeline.run_morphing(
        src_img=src_img,
        tar_img=tar_img,
        morphing_params={**cache_params, "save_cache_path": str(src_cache_dir)},
        seed=args.seed,
        pipeline_type="1536_cascade",
        return_latent=True,
    )
    _, _, _, src_lr_slat = latent
    src_meshes, _ = pipeline.decode_shape_slat(src_lr_slat, 512)
    save_frame(src_meshes[0], 0)
    del latent, src_lr_slat, src_meshes
    cleanup_cuda()

    print("Building target cache...", flush=True)
    _, latent = pipeline.run_morphing(
        src_img=tar_img,
        tar_img=src_img,
        morphing_params={**cache_params, "save_cache_path": str(tar_cache_dir)},
        seed=args.seed,
        pipeline_type="1536_cascade",
        return_latent=True,
    )
    _, _, _, tar_lr_slat = latent
    tar_meshes, _ = pipeline.decode_shape_slat(tar_lr_slat, 512)
    save_frame(tar_meshes[0], args.morphing_num - 1)
    del latent, tar_lr_slat, tar_meshes
    cleanup_cuda()

    morphing_params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": str(src_cache_dir),
        "tar_load_cache_path": str(tar_cache_dir),
        "save_cache_path": str(morph_cache_dir),
        "init_morphing_flag": args.init_morphing,
        "ss_mca_flag": True,
        "ss_tfsa_flag": True,
        "shape_mca_flag": True,
        "shape_tfsa_flag": True,
        "tex_mca_flag": True,
        "tex_tfsa_flag": True,
        "oc_flag": True,
        "tfsa_alpha": 0.8,
        "pipeline_type": "1536_cascade",
        "return_shape_lr": True,
    }

    alpha_array = torch.linspace(1.0, 0.0, args.morphing_num).tolist()
    for morphing_idx in range(1, args.morphing_num - 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        print(f"[{morphing_idx + 1}/{args.morphing_num}] alpha={morphing_params['alpha']:.4f}", flush=True)
        _, latent = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=args.seed,
            pipeline_type="1536_cascade",
            return_latent=True,
        )
        _, _, _, lr_shape_slat = latent
        lr_meshes, _ = pipeline.decode_shape_slat(lr_shape_slat, 512)
        save_frame(lr_meshes[0], morphing_idx)
        del latent, lr_shape_slat, lr_meshes
        cleanup_cuda()

        old_files = list(morph_cache_dir.glob(f"ss_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(morph_cache_dir.glob(f"shape_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        old_files += list(morph_cache_dir.glob(f"tex_sa_morphing{morphing_params['tfsa_cache_idx']}_*"))
        for file_path in old_files:
            file_path.unlink(missing_ok=True)

    if cleanup_cache_root:
        shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
