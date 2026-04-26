import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate product images with OpenAI GPT-Image-1.5 and save them with name_index numbering."
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
        "--model",
        default="gpt-image-1.5",
        help="OpenAI image model name.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        choices=["1024x1024", "1536x1024", "1024x1536", "auto"],
        help="Image size for generation.",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=["low", "medium", "high", "auto"],
        help="Image quality for generation.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "jpeg", "webp"],
        help="Output image format.",
    )
    parser.add_argument(
        "--background",
        default="opaque",
        choices=["opaque", "transparent", "auto"],
        help="Output background mode.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep time between requests to reduce burst traffic.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count per image when the API call fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate jobs and print the planned filenames without calling the API.",
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
        count = job.get("count", 1)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Job #{idx} is missing a valid 'name'.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Job #{idx} is missing a valid 'prompt'.")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"Job #{idx} has invalid 'count'; expected integer >= 1.")
        normalized_jobs.append(
            {
                "name": slugify(name),
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


def next_index_for_name(output_dir: Path, name: str, extension: str) -> int:
    pattern = re.compile(rf"^{re.escape(name)}_(\d{{3}})\.{re.escape(extension)}$")
    max_index = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def save_image_from_result(result, output_path: Path) -> None:
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_path.write_bytes(image_bytes)


def generate_one_image(
    client: OpenAI,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    image_format: str,
    background: str,
):
    return client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=image_format,
        background=background,
    )


def main() -> int:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    jobs_file = Path(args.jobs_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(jobs_file)
    client = OpenAI()

    total_images = sum(job["count"] for job in jobs)
    print(f"Loaded {len(jobs)} job(s), planning {total_images} image(s).")

    for job in jobs:
        name = job["name"]
        prompt = job["prompt"]
        count = job["count"]
        next_index = next_index_for_name(output_dir, name, args.format)

        for offset in range(count):
            image_index = next_index + offset
            output_path = output_dir / f"{name}_{image_index:03d}.{args.format}"

            if args.dry_run:
                print(f"[dry-run] {output_path}")
                continue

            print(f"Generating {output_path.name}")

            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = generate_one_image(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        size=args.size,
                        quality=args.quality,
                        image_format=args.format,
                        background=args.background,
                    )
                    save_image_from_result(result, output_path)
                    print(f"Saved {output_path}")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"Attempt {attempt}/{args.max_retries} failed for {output_path.name}: {exc}",
                        file=sys.stderr,
                    )
                    if attempt < args.max_retries:
                        time.sleep(max(1.0, args.sleep_seconds))

            if last_error is not None:
                print(f"Failed to generate {output_path.name}", file=sys.stderr)
                return 2

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Done. Images saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
