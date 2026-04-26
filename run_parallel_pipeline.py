import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local image generation and TRELLIS 3D generation in parallel."
    )
    parser.add_argument("--local-model-dir", required=True)
    parser.add_argument("--jobs-file", default="inputs/gpt_image_jobs.json")
    parser.add_argument("--input-dir", default="inputs/matong")
    parser.add_argument("--output-dir", default="outputs/matong")
    parser.add_argument("--image-gpu", default="3")
    parser.add_argument("--trellis-gpu", default="2")
    parser.add_argument("--image-env", default="gpt-image-gen-offline")
    parser.add_argument("--trellis-env", default="trellis2")
    parser.add_argument("--image-extra-args", nargs="*", default=["--cpu-offload"])
    parser.add_argument("--trellis-extra-args", nargs="*", default=[])
    parser.add_argument("--watcher-idle-exit-seconds", type=float, default=180.0)
    return parser.parse_args()


def spawn_process(command: list[str], extra_env: dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.Popen(command, env=env)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    image_command = [
        "conda",
        "run",
        "-n",
        args.image_env,
        "python",
        str(project_root / "generate_local_images.py"),
        "--model-dir",
        args.local_model_dir,
        "--jobs-file",
        args.jobs_file,
        "--output-dir",
        args.input_dir,
        *args.image_extra_args,
    ]
    trellis_command = [
        "conda",
        "run",
        "-n",
        args.trellis_env,
        "python",
        str(project_root / "watch_and_generate_3d.py"),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--idle-exit-seconds",
        str(args.watcher_idle_exit_seconds),
        *args.trellis_extra_args,
    ]

    image_env = {
        "CUDA_VISIBLE_DEVICES": args.image_gpu,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    trellis_env = {
        "CUDA_VISIBLE_DEVICES": args.trellis_gpu,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    print("Starting image generator:")
    print(" ".join(image_command))
    image_proc = spawn_process(image_command, image_env)

    print("Starting TRELLIS watcher:")
    print(" ".join(trellis_command))
    trellis_proc = spawn_process(trellis_command, trellis_env)

    try:
        image_return = image_proc.wait()
        print(f"Image generator exited with code {image_return}")
        trellis_return = trellis_proc.wait()
        print(f"TRELLIS watcher exited with code {trellis_return}")
        return 0 if image_return == 0 and trellis_return == 0 else 1
    except KeyboardInterrupt:
        print("Stopping child processes...")
        for proc in (image_proc, trellis_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
