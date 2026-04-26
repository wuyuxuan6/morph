import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run generate_local_images.py for each prompt JSON file "
            "and save images into inputs/<object_name>/ directories."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local path to the SDXL model directory.",
    )
    parser.add_argument(
        "--prompts-dir",
        default="inputs/prompts",
        help="Directory containing per-object prompt JSON files.",
    )
    parser.add_argument(
        "--inputs-root",
        default="inputs",
        help="Root directory where per-object image folders will be created.",
    )
    parser.add_argument(
        "--generator-script",
        default="generate_local_images.py",
        help="Path to the existing image generation script.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used when --conda-env is not set.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Optional conda environment name used to run the generator.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional list of prompt file stems to run, for example: bed toilet.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt",
        default=(
            "cropped, cut off, multiple objects, room scene, interior background, "
            "people, text, watermark, logo, extreme perspective, close-up, blurry, low quality"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_base_command(args: argparse.Namespace, generator_script: Path) -> list[str]:
    active_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if args.conda_env and active_conda_env != args.conda_env:
        command = ["conda", "run", "-n", args.conda_env, "python", str(generator_script)]
    else:
        command = [args.python_bin, str(generator_script)]

    command.extend(
        [
            "--model-dir",
            args.model_dir,
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--seed",
            str(args.seed),
            "--negative-prompt",
            args.negative_prompt,
            "--device",
            args.device,
        ]
    )

    if args.cpu_offload:
        command.append("--cpu-offload")
    if args.attention_slicing:
        command.append("--attention-slicing")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    prompts_dir = (project_root / args.prompts_dir).resolve()
    inputs_root = (project_root / args.inputs_root).resolve()
    generator_script = (project_root / args.generator_script).resolve()

    if not prompts_dir.exists():
        print(f"Prompts directory not found: {prompts_dir}", file=sys.stderr)
        return 1
    if not generator_script.exists():
        print(f"Generator script not found: {generator_script}", file=sys.stderr)
        return 1

    selected = set(args.only)
    prompt_files = sorted(prompts_dir.glob("*.json"))
    if selected:
        prompt_files = [path for path in prompt_files if path.stem in selected]

    if not prompt_files:
        print("No prompt JSON files matched.", file=sys.stderr)
        return 1

    base_command = build_base_command(args, generator_script)

    for prompt_file in prompt_files:
        output_dir = inputs_root / prompt_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            *base_command,
            "--jobs-file",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
        ]

        print(f"\n=== Running {prompt_file.name} -> {output_dir} ===")
        print(" ".join(command))
        result = subprocess.run(command, cwd=project_root)
        if result.returncode != 0:
            print(f"Failed on {prompt_file.name} with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\nAll prompt batches completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
