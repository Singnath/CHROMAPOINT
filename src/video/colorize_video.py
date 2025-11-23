# src/video/colorize_video.py
"""
ChromaPoint — Video colorization script (Step 4)

Uses:
  - The trained image colorization U-Net (Step 2 checkpoint)
  - Temporal fusion (Step 4) with dummy optical flow

Current status:
  * Flow mode is "dummy" (no real motion), but the API is RAFT-ready.
  * Once RAFT is integrated in src.utils.flow.compute_flow(mode="raft"),
    this script will automatically start using true optical flow.

Run from project root:

  conda activate vcolor-mac
  python -m src.video.colorize_video \
      --ckpt models/checkpoints_image/unet_ep1.pt \
      --input data/gray_videos/00001_0268.mp4 \
      --output outputs/00001_0268_colorized.mp4 \
      --alpha 0.6 \
      --device mps

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

import torch
from skimage import color as ski_color

# ---------------------------------------------------------------------
# Make sure project root is on sys.path so "src.*" imports work
# when running this file as a module.
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../Chroma Point
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.temporal_fusion import temporal_fuse_ab


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def _get_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_image_model(ckpt_path: str, device: torch.device):
    """
    Load the Step 2 image colorization model from a checkpoint.

    NOTE:
      We assume the U-Net architecture class is defined in:
        src.models.unet_image
      and is named:
        UNetColorizer

      If your class name is different, change the import line below.
    """
    from src.models.unet_color import UNetColor  # <-- adjust name if needed

    model = UNetColor()
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # assume ckpt itself is a state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Color/format helpers
# ---------------------------------------------------------------------
def _L_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Extract L-channel (0..100) from an RGB uint8 image.
    """
    lab = ski_color.rgb2lab(rgb.astype(np.float32) / 255.0)
    L = lab[..., 0].astype(np.float32)
    return L


def _lab_to_rgb_uint8(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """
    Combine L (float32 [H,W]) and ab (float32 [H,W,2]) into an RGB uint8 image.
    """
    lab = np.stack([L, ab[..., 0], ab[..., 1]], axis=-1).astype(np.float32)
    rgb = ski_color.lab2rgb(lab)  # float in [0,1]
    rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_uint8


def _L_to_tensor(L: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert scalar L in [0,100] (H,W) to model input tensor [1,1,H,W]
    in roughly [-1,1] range, matching training.
    """
    L_norm = (L / 50.0) - 1.0  # 0..100 -> -1..1
    t = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)
    return t.float()


def _ab_from_pred(pred_ab: torch.Tensor) -> np.ndarray:
    """
    Convert model output [1,2,H,W] in ~[-1,1] to ab in Lab scale.
    """
    if pred_ab.ndim != 4:
        raise ValueError(f"Expected [1,2,H,W], got {pred_ab.shape}")
    ab = pred_ab[0].permute(1, 2, 0).detach().cpu().numpy()
    ab = ab * 110.0  # reverse normalization from training
    return ab.astype(np.float32)


# ---------------------------------------------------------------------
# Main video colorization loop
# ---------------------------------------------------------------------
def colorize_video(
    ckpt_path: str,
    input_video: str,
    output_video: str,
    alpha: float = 0.6,
    device_str: str = "cpu",
):
    device = _get_device(device_str)
    print(f"[video] Device: {device}")
    print(f"[video] Loading model from: {ckpt_path}")

    model = load_image_model(ckpt_path, device=device)

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    print(f"[video] Input : {input_video}")
    print(f"[video] Output: {output_video}")
    print(f"[video] Resolution: {W}x{H}, FPS: {fps:.2f}, Frames: {frame_count}")

    prev_rgb = None
    prev_ab = None

    pbar = tqdm(total=frame_count, desc="Colorizing", unit="frame")
    with torch.inference_mode():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR->RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Extract L channel
            L = _L_from_rgb(rgb)
            L_tensor = _L_to_tensor(L, device)

            # Predict ab for current frame
            pred_ab_norm = model(L_tensor)  # [1,2,H,W]
            ab_pred = _ab_from_pred(pred_ab_norm)  # [H,W,2]

            # Temporal fusion
            if prev_rgb is None or prev_ab is None:
                ab_fused = ab_pred
            else:
                ab_fused_t, _dbg = temporal_fuse_ab(
                    prev_rgb,
                    rgb,
                    prev_ab,
                    ab_pred,
                    device=device,
                    alpha=alpha,
                    flow_mode="dummy",  # later: "raft" on GPU
                    debug=False,
                )
                ab_fused = (
                    ab_fused_t[0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            # Reconstruct RGB
            rgb_out = _lab_to_rgb_uint8(L, ab_fused)
            bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
            writer.write(bgr_out)

            # Update temporal state
            prev_rgb = rgb
            prev_ab = ab_fused

            pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print("[video] Done! Saved:", output_video)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="ChromaPoint: video colorization with temporal fusion"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to image colorization checkpoint (.pt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video (grayscale or near-grayscale mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save colorized mp4",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Temporal blend weight (0 = only current frame, 1 = only warped previous)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device: 'cpu', 'mps', 'cuda', or 'cuda:0', etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    colorize_video(
        ckpt_path=args.ckpt,
        input_video=args.input,
        output_video=args.output,
        alpha=args.alpha,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()