import argparse
import os

from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import run_morphing, run_morphing_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="assets/example_image/T.png")
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--pipeline-type", default=os.environ.get("TRELLIS2_PIPELINE_TYPE", "1024"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morphing-num", type=int, default=24)
    parser.add_argument("--output-dir", default="outputs/3Dmorphing")
    parser.add_argument("--enable-tfsa", action="store_true")
    parser.add_argument("--disable-oc", action="store_true")
    parser.add_argument("--init-morphing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    src_name = os.path.splitext(os.path.basename(args.src))[0]
    tar_name = os.path.splitext(os.path.basename(args.tar))[0]

    src_cache_dir = os.path.join("outputs", "cache", src_name, "cache")
    tar_cache_dir = os.path.join("outputs", "cache", tar_name, "cache")
    os.makedirs(src_cache_dir, exist_ok=True)
    os.makedirs(tar_cache_dir, exist_ok=True)

    cache_params = {
        "save_cache_path": src_cache_dir,
        "pipeline_type": args.pipeline_type,
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "ss_tfsa_flag": False,
        "shape_mca_flag": False,
        "shape_tfsa_flag": False,
        "tex_mca_flag": False,
        "tex_tfsa_flag": False,
    }
    if not os.path.exists(f"{src_cache_dir}/tex_slat_init.pt"):
        run_morphing_cache(
            pipeline,
            src_img,
            tar_img,
            {**cache_params, "save_cache_path": src_cache_dir},
            args.seed,
            os.path.dirname(src_cache_dir),
            src_name,
            render_kwargs={"verbose": False},
        )
    if not os.path.exists(f"{tar_cache_dir}/tex_slat_init.pt"):
        run_morphing_cache(
            pipeline,
            tar_img,
            src_img,
            {**cache_params, "save_cache_path": tar_cache_dir},
            args.seed,
            os.path.dirname(tar_cache_dir),
            tar_name,
            render_kwargs={"verbose": False},
        )

    name = f"{src_name}+{tar_name}"
    save_path = os.path.join(args.output_dir, name)
    save_cache_path = os.path.join(save_path, "cache")
    os.makedirs(save_cache_path, exist_ok=True)

    morphing_params = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": src_cache_dir,
        "tar_load_cache_path": tar_cache_dir,
        "save_cache_path": save_cache_path,
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

    run_morphing(
        pipeline,
        src_img,
        tar_img,
        morphing_params,
        args.seed,
        save_path,
        name,
        render_kwargs={"verbose": False},
    )


if __name__ == "__main__":
    main()
