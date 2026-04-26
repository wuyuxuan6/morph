import os
import argparse
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import torch
import imageio.v2 as imageio

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import get_default_envmap, ROT_VIDEO_RESOLUTION, ROT_VIDEO_NUM_FRAMES
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
    parser = argparse.ArgumentParser(description="Pure single-image Trellis2 generation with optional legacy CFG formula.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-resolution", type=int, default=512)
    parser.add_argument("--video-resolution", type=int, default=ROT_VIDEO_RESOLUTION)
    parser.add_argument("--video-num-frames", type=int, default=ROT_VIDEO_NUM_FRAMES)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=500000)
    parser.add_argument("--legacy-cfg-formula", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.model_path}", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    envmap = get_default_envmap(device=f"cuda:{args.gpu}")
    print("Pipeline loaded and moved to CUDA", flush=True)

    image = Image.open(args.image)
    stem = Path(args.image).stem

    common_sampler_kwargs = {"legacy_cfg_formula": args.legacy_cfg_formula}
    outputs = pipeline.run(
        image=image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        sparse_structure_sampler_params=common_sampler_kwargs,
        shape_slat_sampler_params=common_sampler_kwargs,
        tex_slat_sampler_params=common_sampler_kwargs,
    )
    mesh = outputs[0]

    print("Saving preview png", flush=True)
    snapshot = render_utils.render_snapshot(mesh, resolution=args.preview_resolution, nviews=4, envmap=envmap)
    if "shaded" in snapshot:
        frames = render_utils.make_pbr_vis_frames(snapshot, resolution=args.preview_resolution)
        Image.fromarray(frames[0]).save(output_dir / f"{stem}.png")
    elif "color" in snapshot:
        Image.fromarray(snapshot["color"][0]).save(output_dir / f"{stem}.png")
    else:
        first_key = next(iter(snapshot))
        Image.fromarray(snapshot[first_key][0]).save(output_dir / f"{stem}.png")

    print("Saving glb", flush=True)
    export_glb(mesh, output_dir / f"{stem}.glb", texture_size=args.texture_size, decimation_target=args.decimation_target)

    print("Rendering endpoint video", flush=True)
    video = render_utils.render_rot_video(
        mesh,
        resolution=args.video_resolution,
        num_frames=args.video_num_frames,
        envmap=envmap,
    )
    frames = render_utils.make_pbr_vis_frames(video, resolution=min(args.video_resolution, 1024))
    imageio.mimsave(output_dir / f"{stem}.mp4", frames, fps=15)


if __name__ == "__main__":
    main()
