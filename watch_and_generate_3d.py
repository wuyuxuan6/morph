import argparse
import os
import time
import traceback
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        description="Watch an input folder and generate 3D assets for new images."
    )
    parser.add_argument("--input-dir", default="inputs/matong")
    parser.add_argument("--output-dir", default="outputs/matong")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--pipeline-type",
        default=None,
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=1000000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--envmap", default="assets/hdri/forest.exr")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=0.0,
        help="Exit after being caught up for this many seconds. 0 means run forever.",
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


def candidate_images(input_dir: Path):
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    envmap = build_envmap(args.envmap) if args.render_video else None

    print("Loading TRELLIS pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print("TRELLIS pipeline loaded.")

    stable_observations = {}
    last_progress_at = time.time()

    while True:
        pending = []
        now = time.time()

        for image_path in candidate_images(input_dir):
            stem = image_path.stem
            glb_path = output_dir / f"{stem}.glb"
            if glb_path.exists():
                continue

            stat = image_path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            previous = stable_observations.get(image_path)
            stable_count = 1 if previous is None or previous[0] != signature else previous[1] + 1
            stable_observations[image_path] = (signature, stable_count, now)

            age = now - stat.st_mtime
            if stable_count >= 2 and age >= args.settle_seconds:
                pending.append(image_path)

        if pending:
            for image_path in pending:
                stem = image_path.stem
                glb_path = output_dir / f"{stem}.glb"
                mp4_path = output_dir / f"{stem}.mp4"
                failed_marker = output_dir / f"{stem}.failed.txt"
                try:
                    print(f"Processing new image: {image_path}")
                    image = Image.open(image_path)
                    mesh = pipeline.run(
                        image,
                        seed=args.seed,
                        pipeline_type=args.pipeline_type,
                    )[0]
                    mesh.simplify(16777216)

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
                    if failed_marker.exists():
                        failed_marker.unlink()
                    last_progress_at = time.time()
                    print(f"Finished {image_path.name}")
                except Exception as exc:
                    failed_marker.write_text(
                        f"Image: {image_path}\nError: {exc}\n\n{traceback.format_exc()}",
                        encoding="utf-8",
                    )
                    print(f"Failed {image_path.name}: {exc}")
                finally:
                    torch.cuda.empty_cache()
                    stable_observations.pop(image_path, None)
        else:
            if args.idle_exit_seconds > 0 and now - last_progress_at >= args.idle_exit_seconds:
                print("No new images detected within idle window. Exiting watcher.")
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
