import os
import random
from glob import glob

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

from ..modules.sparse import SparseTensor
from . import render_utils
from ..renderers import EnvMap


ROT_VIDEO_RESOLUTION = int(os.environ.get("TRELLIS2_RENDER_RESOLUTION", "1024"))
ROT_VIDEO_NUM_FRAMES = int(os.environ.get("TRELLIS2_RENDER_NUM_FRAMES", "120"))


def get_default_envmap(device: str = "cuda"):
    envmap_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "hdri", "forest.exr")
    if not os.path.exists(envmap_path):
        raise FileNotFoundError(f"Default envmap not found: {envmap_path}")
    image = cv2.cvtColor(cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(image, dtype=torch.float32, device=device))


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_rows_with_mask(x: torch.Tensor) -> torch.Tensor:
    seen = set()
    mask = []
    for row in x.tolist():
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            mask.append(True)
        else:
            mask.append(False)
    return torch.tensor(mask, dtype=torch.bool, device=x.device)


def cal_cossim_matrix(feats1: torch.Tensor, feats2: torch.Tensor) -> torch.Tensor:
    feats1 = F.normalize(feats1.reshape(feats1.shape[0], -1), dim=-1)
    feats2 = F.normalize(feats2.reshape(feats2.shape[0], -1), dim=-1)
    return feats1 @ feats2.t()


def cal_eucdist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    xx = (tensor1 * tensor1).sum(dim=1, keepdim=True)
    yy = (tensor2 * tensor2).sum(dim=1, keepdim=True).t()
    d2 = xx + yy - 2.0 * (tensor1 @ tensor2.t())
    d2.clamp_(min=0)
    return torch.sqrt(d2)


def feature_interp(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    alpha: float,
    mapping_mode: str = "order",
    interp_mode: str = "linear",
    indices=None,
) -> torch.Tensor:
    if mapping_mode == "cossim":
        indices = torch.argmax(cal_cossim_matrix(tensor1, tensor2), dim=1)
        tensor2 = tensor2[indices]
    elif mapping_mode == "hungarian":
        tensor2 = tensor2[indices]

    if interp_mode == "linear":
        return alpha * tensor1 + (1 - alpha) * tensor2
    if interp_mode == "slerp":
        a = tensor1
        b = tensor2
        an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        bn = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dot = (an * bn).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        close = (dot.abs() > 0.9999).squeeze(-1)
        theta = torch.acos(dot) * (1 - alpha)
        rel = bn - dot * an
        rel = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        slerp_dir = an * torch.cos(theta) + rel * torch.sin(theta)
        lin_dir = alpha * an + (1 - alpha) * bn
        lin_dir = lin_dir / lin_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        slerp_dir = torch.where(close.unsqueeze(-1), lin_dir, slerp_dir)
        ra = a.norm(dim=-1, keepdim=True)
        rb = b.norm(dim=-1, keepdim=True)
        r = alpha * ra + (1 - alpha) * rb
        return slerp_dir * r
    raise ValueError(f"Unsupported interp_mode: {interp_mode}")


def slat_interp(
    slat1: SparseTensor,
    slat2: SparseTensor,
    alpha: float,
    mapping_mode: str = "order",
    interp_mode: str = "linear",
    unique_flag: bool = True,
    indices=None,
) -> SparseTensor:
    if mapping_mode == "order":
        if slat1.feats.shape[0] <= slat2.feats.shape[0]:
            indices = torch.arange(slat1.feats.shape[0], device=slat1.feats.device)
        else:
            times = slat1.feats.shape[0] // slat2.feats.shape[0]
            rem = slat1.feats.shape[0] % slat2.feats.shape[0]
            pieces = [torch.arange(slat2.feats.shape[0], device=slat1.feats.device)] * times
            if rem:
                pieces.append(torch.arange(rem, device=slat1.feats.device))
            indices = torch.cat(pieces, dim=0)
    elif mapping_mode == "cossim":
        indices = torch.argmax(cal_cossim_matrix(slat1.feats, slat2.feats), dim=1)
    elif mapping_mode == "eucdist":
        indices = torch.argmin(cal_eucdist_matrix(slat1.coords[:, 1:].float(), slat2.coords[:, 1:].float()), dim=1)

    interp_feats = feature_interp(slat1.feats, slat2.feats[indices], alpha, interp_mode=interp_mode)
    interp_coords = torch.round(alpha * slat1.coords + (1 - alpha) * slat2.coords[indices]).int()
    if unique_flag:
        mask = unique_rows_with_mask(interp_coords)
        interp_feats = interp_feats[mask]
        interp_coords = interp_coords[mask]
    return SparseTensor(feats=interp_feats, coords=interp_coords)


def rotate_pc(
    points: torch.Tensor,
    angle: float,
    axis: str = "z",
) -> torch.Tensor:
    single = points.dim() == 2
    if single:
        points = points.unsqueeze(0)
    bsz, _, dims = points.shape
    if dims != 3:
        raise ValueError(f"Expected 3D points, got shape {points.shape}")

    device, dtype = points.device, points.dtype
    ang = torch.as_tensor(angle, device=device, dtype=dtype) * torch.pi / 180.0
    if ang.dim() == 0:
        ang = ang.expand(bsz)
    c = torch.cos(ang)
    s = torch.sin(ang)

    if axis in ("x", 0):
        rot = torch.stack([
            torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), c, -s], dim=-1),
            torch.stack([torch.zeros_like(c), s, c], dim=-1),
        ], dim=-2)
    elif axis in ("y", 1):
        rot = torch.stack([
            torch.stack([c, torch.zeros_like(c), s], dim=-1),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([-s, torch.zeros_like(c), c], dim=-1),
        ], dim=-2)
    elif axis in ("z", 2):
        rot = torch.stack([
            torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
            torch.stack([s, c, torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1),
        ], dim=-2)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

    rotated = torch.matmul(points, rot.transpose(1, 2))
    return rotated.squeeze(0) if single else rotated


def chamfer_distance_bidirectional(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float("inf"), device=points1.device if points1.numel() > 0 else points2.device)
    dist = torch.cdist(points1, points2, p=2)
    return dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()


def get_sparse_stage(kwargs: dict) -> str:
    return kwargs.get("morph_stage", "slat")


def get_sparse_mca_flag(kwargs: dict) -> bool:
    stage = get_sparse_stage(kwargs)
    if f"{stage}_mca_flag" in kwargs:
        return kwargs[f"{stage}_mca_flag"]
    if stage.startswith("shape"):
        return kwargs.get("shape_mca_flag", kwargs.get("slat_mca_flag", False))
    if stage.startswith("tex"):
        return kwargs.get("tex_mca_flag", kwargs.get("slat_mca_flag", False))
    return kwargs.get("slat_mca_flag", False)


def get_sparse_tfsa_flag(kwargs: dict) -> bool:
    stage = get_sparse_stage(kwargs)
    if f"{stage}_tfsa_flag" in kwargs:
        return kwargs[f"{stage}_tfsa_flag"]
    if stage.startswith("shape"):
        return kwargs.get("shape_tfsa_flag", kwargs.get("slat_tfsa_flag", False))
    if stage.startswith("tex"):
        return kwargs.get("tex_tfsa_flag", kwargs.get("slat_tfsa_flag", False))
    return kwargs.get("slat_tfsa_flag", False)


def get_sparse_cache_prefix(kwargs: dict) -> str:
    return get_sparse_stage(kwargs)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def image_to_frame(image, resolution: int) -> np.ndarray:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4])
    frame = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(frame).resize((resolution, resolution), Image.Resampling.LANCZOS))


def find_best_view_idx(frames: list[np.ndarray], ref_image: Image.Image) -> int:
    if not frames:
        return 0
    ref = image_to_frame(ref_image, frames[0].shape[0]).astype(np.float32)
    scores = [((frame.astype(np.float32) - ref) ** 2).mean() for frame in frames]
    return int(np.argmin(scores))


def flip_view_idx(view_idx: int, num_frames: int) -> int:
    if num_frames <= 0:
        return view_idx
    return (view_idx + num_frames // 2) % num_frames


def render_single_view_frame(pipeline, image, seed: int, pipeline_type: str, view_idx: int, render_kwargs: dict) -> np.ndarray:
    outputs = pipeline.run(
        image=image,
        seed=seed,
        pipeline_type=pipeline_type,
    )
    video = render_utils.render_rot_video(
        outputs[0],
        resolution=ROT_VIDEO_RESOLUTION,
        num_frames=ROT_VIDEO_NUM_FRAMES,
        **render_kwargs,
    )
    frame = video["shaded"][view_idx]
    del outputs
    cleanup_cuda()
    return frame


def run_morphing_cache(pipeline, src_img, tar_img, morphing_params, seed, save_path, name, render_kwargs=None):
    render_kwargs = render_kwargs or {}
    if "envmap" not in render_kwargs:
        render_kwargs["envmap"] = get_default_envmap("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    outputs = pipeline.run_morphing(
        src_img=src_img,
        tar_img=tar_img,
        morphing_params=morphing_params,
        seed=seed,
        pipeline_type=morphing_params.get("pipeline_type"),
    )
    video = render_utils.render_rot_video(
        outputs[0],
        resolution=ROT_VIDEO_RESOLUTION,
        num_frames=ROT_VIDEO_NUM_FRAMES,
        **render_kwargs,
    )
    frames = render_utils.make_pbr_vis_frames(video, resolution=min(ROT_VIDEO_RESOLUTION, 1024))
    imageio.mimsave(f"{save_path}/{name}.mp4", frames, fps=15)
    del outputs
    cleanup_cuda()


def run_morphing(pipeline, src_img, tar_img, morphing_params, seed, save_path, name, render_kwargs=None):
    render_kwargs = render_kwargs or {}
    if "envmap" not in render_kwargs:
        render_kwargs["envmap"] = get_default_envmap("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    files = glob(f"{morphing_params['save_cache_path']}/*")
    for f in files:
        os.remove(f)

    if os.path.exists(f"{save_path}/morphing_{name}.mp4"):
        print(f"Skip existing {name}")
        return

    frames = []
    single_view_frames = []
    single_view_idx = None
    src_ref = pipeline.preprocess_image(src_img.copy()) if hasattr(pipeline, "preprocess_image") else src_img
    tar_ref = pipeline.preprocess_image(tar_img.copy()) if hasattr(pipeline, "preprocess_image") else tar_img
    alpha_array = np.linspace(1, 0, morphing_params["morphing_num"])
    for morphing_idx in range(1, morphing_params["morphing_num"] - 1):
        morphing_params["alpha"] = float(alpha_array[morphing_idx])
        morphing_params["morphing_idx"] = morphing_idx
        morphing_params["tfsa_cache_idx"] = morphing_idx - 1
        morphing_params["tfsa_alpha"] = morphing_params.get("tfsa_alpha", 0.8)
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=seed,
            pipeline_type=morphing_params.get("pipeline_type"),
        )
        video = render_utils.render_rot_video(
            outputs[0],
            resolution=ROT_VIDEO_RESOLUTION,
            num_frames=ROT_VIDEO_NUM_FRAMES,
            **render_kwargs,
        )
        vis_frames = np.stack(render_utils.make_pbr_vis_frames(video, resolution=min(ROT_VIDEO_RESOLUTION, 1024)), axis=0)
        frames.append(vis_frames)
        shaded_frames = video["shaded"]
        if single_view_idx is None:
            single_view_idx = flip_view_idx(find_best_view_idx(shaded_frames, src_ref), len(shaded_frames))
        single_view_frames.append(shaded_frames[single_view_idx])
        del outputs
        cleanup_cuda()

        old_files = glob(f"{morphing_params['save_cache_path']}/ss_sa_morphing{morphing_params['tfsa_cache_idx']}_*")
        old_files += glob(f"{morphing_params['save_cache_path']}/shape_sa_morphing{morphing_params['tfsa_cache_idx']}_*")
        old_files += glob(f"{morphing_params['save_cache_path']}/tex_sa_morphing{morphing_params['tfsa_cache_idx']}_*")
        for f in old_files:
            os.remove(f)

    files = glob(f"{morphing_params['save_cache_path']}/*")
    for f in files:
        os.remove(f)

    if frames:
        montage = np.concatenate(frames, axis=2)
        imageio.mimsave(f"{save_path}/morphing_{name}.mp4", montage, fps=max(1, (morphing_params["morphing_num"] - 2) // 2))
    if single_view_frames:
        pipeline_type = morphing_params.get("pipeline_type")
        src_frame = render_single_view_frame(pipeline, src_img, seed, pipeline_type, single_view_idx, render_kwargs)
        tar_frame = render_single_view_frame(pipeline, tar_img, seed, pipeline_type, single_view_idx, render_kwargs)
        single_view = np.stack([src_frame] + single_view_frames + [tar_frame], axis=0)
        imageio.mimsave(
            f"{save_path}/morphing_{name}_singleview.mp4",
            single_view,
            fps=max(1, (morphing_params["morphing_num"] - 2) // 2),
        )
