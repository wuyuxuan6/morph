import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run batch_example.py or watch_and_generate_3d.py for "
            "multiple per-object input directories."
        )
    )
    parser.add_argument(
        "--mode",
        default="batch",
        choices=["batch", "watch"],
        help="Use batch_example.py or watch_and_generate_3d.py.",
    )
    parser.add_argument(
        "--inputs-root",
        default="inputs",
        help="Root directory containing per-object image folders.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing per-object 3D output folders.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=["bed", "carpet", "toilet", "bedside_table", "wardrobe", "wash_basin"],
        help="Object folder names to process.",
    )
    parser.add_argument(
        "--batch-script",
        default="batch_example.py",
        help="Path to the batch image-to-3D script.",
    )
    parser.add_argument(
        "--watch-script",
        default="watch_and_generate_3d.py",
        help="Path to the watcher image-to-3D script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used when --conda-env is not set.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Optional conda environment name used to run the 3D script.",
    )
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
    parser.add_argument("--idle-exit-seconds", type=float, default=180.0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def base_command(args: argparse.Namespace, script_path: Path) -> list[str]:
    active_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if args.conda_env and active_conda_env != args.conda_env:
        command = ["conda", "run", "-n", args.conda_env, "python", str(script_path)]
    else:
        command = [args.python_bin, str(script_path)]

    command.extend(
        [
            "--model",
            args.model,
            "--seed",
            str(args.seed),
            "--decimation-target",
            str(args.decimation_target),
            "--texture-size",
            str(args.texture_size),
            "--envmap",
            args.envmap,
        ]
    )
    if args.pipeline_type:
        command.extend(["--pipeline-type", args.pipeline_type])
    if args.render_video:
        command.append("--render-video")
    if args.mode == "watch":
        command.extend(
            [
                "--poll-interval",
                str(args.poll_interval),
                "--settle-seconds",
                str(args.settle_seconds),
                "--idle-exit-seconds",
                str(args.idle_exit_seconds),
            ]
        )
    return command


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    inputs_root = (project_root / args.inputs_root).resolve()
    outputs_root = (project_root / args.outputs_root).resolve()
    script_name = args.batch_script if args.mode == "batch" else args.watch_script
    script_path = (project_root / script_name).resolve()

    if not script_path.exists():
        print(f"3D script not found: {script_path}", file=sys.stderr)
        return 1

    command_prefix = base_command(args, script_path)

    for obj in args.objects:
        input_dir = inputs_root / obj
        output_dir = outputs_root / obj

        if not input_dir.exists():
            message = f"Input directory not found: {input_dir}"
            if args.skip_missing:
                print(f"Skipping. {message}")
                continue
            print(message, file=sys.stderr)
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *command_prefix,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]

        print(f"\n=== Running {obj}: {input_dir} -> {output_dir} ===")
        print(" ".join(command))
        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=project_root)
        if result.returncode != 0:
            print(f"Failed on {obj} with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\nAll 3D batches completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
