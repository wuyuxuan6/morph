import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import StableDiffusionXLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local SDXL images with name_index numbering."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local path to the SDXL model directory.",
    )
    parser.add_argument(
        "--jobs-file",
        default="inputs/gpt_image_jobs.json",
        help="Path to a JSON file containing generation jobs.",
    )
    parser.add_argument(
        "--output-dir",
        default="inputs/matong",
        help="Directory used to store generated images.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps per image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "cropped, cut off, multiple objects, room scene, interior background, "
            "people, text, watermark, logo, extreme perspective, close-up, blurry, low quality"
        ),
        help="Negative prompt used for all jobs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable model CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
        help="Enable attention slicing to reduce VRAM usage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate jobs and print planned filenames without generating images.",
    )
    return parser.parse_args()


def load_jobs(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Jobs file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Jobs file must contain a non-empty JSON array.")

    normalized_jobs = []
    for idx, job in enumerate(data, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Job #{idx} must be an object.")
        name = job.get("name")
        prompt = job.get("prompt")
        prompts = job.get("prompts")
        count = job.get("count", 1)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Job #{idx} is missing a valid 'name'.")
        normalized_name = slugify(name)
        if prompts is not None:
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(f"Job #{idx} has invalid 'prompts'; expected non-empty array.")
            cleaned_prompts = []
            for prompt_idx, prompt_item in enumerate(prompts, start=1):
                if not isinstance(prompt_item, str) or not prompt_item.strip():
                    raise ValueError(
                        f"Job #{idx} prompt #{prompt_idx} is invalid; expected non-empty string."
                    )
                cleaned_prompts.append(prompt_item.strip())
            normalized_jobs.append(
                {
                    "name": normalized_name,
                    "prompts": cleaned_prompts,
                    "count": len(cleaned_prompts),
                }
            )
        else:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Job #{idx} is missing a valid 'prompt'.")
            if not isinstance(count, int) or count < 1:
                raise ValueError(f"Job #{idx} has invalid 'count'; expected integer >= 1.")
            normalized_jobs.append(
                {
                    "name": normalized_name,
                    "prompt": prompt.strip(),
                    "count": count,
                }
            )
    return normalized_jobs


def slugify(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Job name becomes empty after normalization.")
    return value


def next_index_for_name(output_dir: Path, name: str) -> int:
    pattern = re.compile(rf"^{re.escape(name)}_(\d{{3}})\.png$")
    max_index = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def load_pipeline(model_dir: Path, device: str, cpu_offload: bool, attention_slicing: bool):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    if attention_slicing:
        pipeline.enable_attention_slicing()
    if device == "cuda":
        if cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to("cpu")
    return pipeline


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    jobs_file = Path(args.jobs_file)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(jobs_file)

    total_images = sum(job["count"] for job in jobs)
    print(f"Loaded {len(jobs)} job(s), planning {total_images} image(s).")

    if args.dry_run:
        for job in jobs:
            next_index = next_index_for_name(output_dir, job["name"])
            for offset in range(job["count"]):
                output_path = output_dir / f"{job['name']}_{next_index + offset:03d}.png"
                print(f"[dry-run] {output_path}")
        print(f"Done. Images saved to {output_dir}")
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available.", file=sys.stderr)
        return 2

    pipeline = load_pipeline(
        model_dir=model_dir,
        device=args.device,
        cpu_offload=args.cpu_offload,
        attention_slicing=args.attention_slicing,
    )

    for job in jobs:
        next_index = next_index_for_name(output_dir, job["name"])
        for offset in range(job["count"]):
            image_index = next_index + offset
            output_path = output_dir / f"{job['name']}_{image_index:03d}.png"
            prompt = job["prompts"][offset] if "prompts" in job else job["prompt"]
            seed = args.seed + image_index - 1
            generator = (
                torch.Generator(device=args.device).manual_seed(seed)
                if args.device == "cuda"
                else torch.Generator().manual_seed(seed)
            )

            print(f"Generating {output_path.name} with seed {seed}")
            image = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
            image.save(output_path)
            print(f"Saved {output_path}")

    print(f"Done. Images saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
