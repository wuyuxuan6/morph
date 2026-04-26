import argparse
import os

from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils.morphing_utils import run_morphing, run_morphing_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--model-path", default=os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--pipeline-type", default=os.environ.get("TRELLIS2_PIPELINE_TYPE", "512"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--morphing-num", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    return parser.parse_args()


def ensure_cache(pipeline, img_a, img_b, name, pipeline_type, seed):
    save_path = os.path.join("outputs", "cache", name)
    save_cache_path = os.path.join(save_path, "cache")
    os.makedirs(save_cache_path, exist_ok=True)
    params = {
        "save_cache_path": save_cache_path,
        "pipeline_type": pipeline_type,
        "init_morphing_flag": False,
        "ss_mca_flag": False,
        "ss_tfsa_flag": False,
        "shape_mca_flag": False,
        "shape_tfsa_flag": False,
        "tex_mca_flag": False,
        "tex_tfsa_flag": False,
        "oc_flag": False,
    }
    if not os.path.exists(f"{save_cache_path}/tex_slat_init.pt"):
        run_morphing_cache(
            pipeline,
            img_a,
            img_b,
            params,
            seed,
            save_path,
            name,
            render_kwargs={"verbose": False},
        )
    return save_cache_path


def case_params(base, *, ss, shape, tex, oc=True):
    params = dict(base)
    params.update({
        "init_morphing_flag": False,
        "ss_mca_flag": ss,
        "ss_tfsa_flag": ss,
        "shape_mca_flag": shape,
        "shape_tfsa_flag": shape,
        "tex_mca_flag": tex,
        "tex_tfsa_flag": tex,
        "oc_flag": oc,
        "tfsa_alpha": 0.8,
    })
    return params


def case_params_split(base, *, ss_mca, ss_tfsa, shape_mca, shape_tfsa, tex_mca, tex_tfsa, oc):
    params = dict(base)
    params.update({
        "init_morphing_flag": False,
        "ss_mca_flag": ss_mca,
        "ss_tfsa_flag": ss_tfsa,
        "shape_mca_flag": shape_mca,
        "shape_tfsa_flag": shape_tfsa,
        "tex_mca_flag": tex_mca,
        "tex_tfsa_flag": tex_tfsa,
        "oc_flag": oc,
        "tfsa_alpha": 0.8,
    })
    return params


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    src_img = Image.open(args.src)
    tar_img = Image.open(args.tar)
    src_name = os.path.splitext(os.path.basename(args.src))[0]
    tar_name = os.path.splitext(os.path.basename(args.tar))[0]

    src_cache = ensure_cache(pipeline, src_img, tar_img, src_name, args.pipeline_type, args.seed)
    tar_cache = ensure_cache(pipeline, tar_img, src_img, tar_name, args.pipeline_type, args.seed)

    pair_name = f"{src_name}+{tar_name}"
    base = {
        "morphing_num": args.morphing_num,
        "src_load_cache_path": src_cache,
        "tar_load_cache_path": tar_cache,
        "pipeline_type": args.pipeline_type,
    }

    cases = [
        ("self_src_src_plain", src_img, src_img, case_params_split(base, ss_mca=False, ss_tfsa=False, shape_mca=False, shape_tfsa=False, tex_mca=False, tex_tfsa=False, oc=False)),
        ("self_src_src_no_temporal", src_img, src_img, case_params_split(base, ss_mca=True, ss_tfsa=False, shape_mca=True, shape_tfsa=False, tex_mca=True, tex_tfsa=False, oc=False)),
        ("self_src_src", src_img, src_img, case_params(base, ss=True, shape=True, tex=True)),
        ("self_tar_tar", tar_img, tar_img, case_params({
            **base,
            "src_load_cache_path": tar_cache,
            "tar_load_cache_path": tar_cache,
        }, ss=True, shape=True, tex=True)),
        ("full", src_img, tar_img, case_params(base, ss=True, shape=True, tex=True)),
        ("shape_only", src_img, tar_img, case_params(base, ss=True, shape=True, tex=False)),
        ("tex_only", src_img, tar_img, case_params(base, ss=False, shape=False, tex=True, oc=False)),
    ]

    for case_name, case_src, case_tar, params in cases:
        save_path = os.path.join(args.output_dir, pair_name, case_name)
        save_cache_path = os.path.join(save_path, "cache")
        os.makedirs(save_cache_path, exist_ok=True)
        params = {**params, "save_cache_path": save_cache_path}
        run_morphing(
            pipeline,
            case_src,
            case_tar,
            params,
            args.seed,
            save_path,
            case_name,
            render_kwargs={"verbose": False},
        )


if __name__ == "__main__":
    main()
