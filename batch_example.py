import argparse
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

import cv2
import imageio
import o_voxel
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch image-to-3D generation based on example.py."
    )
    parser.add_argument(
        "--input-dir",
        default="inputs/matong",
        help="Directory containing input images. Default: inputs/matong",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/matong",
        help="Directory for generated assets. Default: outputs/matong",
    )
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS.2-4B",
        help="Pretrained model name or local path.",
    )
    parser.add_argument(
        "--pipeline-type",
        default=None,
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Pipeline type. Uses the model default when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for every image. Default: 42",
    )
    parser.add_argument(
        "--decimation-target",
        type=int,
        default=1000000,
        help="Target face count for GLB export. Default: 1000000",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=4096,
        help="Texture size for GLB export. Default: 4096",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Also render an mp4 preview for each image.",
    )
    parser.add_argument(
        "--envmap",
        default="assets/hdri/forest.exr",
        help="HDRI path used when --render-video is enabled.",
    )
    return parser.parse_args()


def build_envmap(envmap_path: str) -> EnvMap:
    envmap_image = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
    if envmap_image is None:
        raise FileNotFoundError(f"Failed to load envmap: {envmap_path}")
    return EnvMap(
        torch.tensor(
            cv2.cvtColor(envmap_image, cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        )
    )


def export_glb(mesh, output_path: Path, decimation_target: int, texture_size: int) -> None:
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


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {input_dir}. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    envmap = build_envmap(args.envmap) if args.render_video else None

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    print(f"Found {len(image_paths)} image(s) in {input_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        glb_path = output_dir / f"{stem}.glb"
        mp4_path = output_dir / f"{stem}.mp4"

        print(f"[{index}/{len(image_paths)}] Processing {image_path}")
        image = Image.open(image_path)
        mesh = pipeline.run(
            image,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )[0]
        mesh.simplify(16777216)  # nvdiffrast limit

        if args.render_video:
            video = render_utils.make_pbr_vis_frames(
                render_utils.render_video(mesh, envmap=envmap)
            )
            imageio.mimsave(mp4_path, video, fps=15)

        export_glb(
            mesh=mesh,
            output_path=glb_path,
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
        )
        torch.cuda.empty_cache()

    print(f"Done. Assets saved to {output_dir}")


if __name__ == "__main__":
    main()
