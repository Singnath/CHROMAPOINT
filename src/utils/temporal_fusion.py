# src/utils/temporal_fusion.py
"""
Temporal fusion utilities for ChromaPoint.

This module takes the per-frame chroma prediction from the image model and
fuses it with warped chroma from the previous frame using optical flow.

Right now we call `compute_flow(..., mode="dummy")` from src.utils.flow,
which returns zero flow. That means the temporal fusion behaves like a simple
EMA over time. Later, on a GPU machine, you can switch mode="raft" to use
true RAFT optical flow without changing the rest of the code.
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np
import torch

from src.utils.flow import compute_flow, warp_with_flow


def _to_ab_tensor(ab, device: str | torch.device) -> torch.Tensor:
    """
    Ensure ab is a float tensor of shape [1, 2, H, W] on the given device.
    ab can be:
      - numpy array [H, W, 2]
      - torch tensor [2, H, W]
      - torch tensor [1, 2, H, W]
    """
    if isinstance(ab, np.ndarray):
        if ab.ndim != 3 or ab.shape[2] != 2:
            raise ValueError(f"Expected ab ndarray [H,W,2], got {ab.shape}")
        t = torch.from_numpy(ab).permute(2, 0, 1).unsqueeze(0)  # 1x2xHxW
    elif isinstance(ab, torch.Tensor):
        t = ab
        if t.ndim == 3 and t.shape[0] == 2:
            t = t.unsqueeze(0)  # 1x2xHxW
        elif t.ndim == 3 and t.shape[-1] == 2:
            t = t.permute(2, 0, 1).unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"Expected [2,H,W] or [1,2,H,W], got {t.shape}")
    else:
        raise TypeError(f"Unsupported ab type: {type(ab)}")

    return t.to(device).float()


def temporal_fuse_ab(
    prev_rgb,
    curr_rgb,
    ab_prev,
    ab_pred,
    device: str | torch.device = "cpu",
    alpha: float = 0.6,
    flow_mode: str = "dummy",
    debug: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Fuse previous-frame chroma with current prediction using optical flow.

    Args:
        prev_rgb: previous RGB frame (HxWx3, uint8/float32 or tensor).
        curr_rgb: current RGB frame  (same type/shape as prev_rgb).
        ab_prev:  previous frame ab in native scale (typically [-110, 110]).
                  Shape [H,W,2], [2,H,W] or [1,2,H,W].
        ab_pred:  current frame predicted ab (same scale & shape as ab_prev).
        device:   "cpu", "mps", or "cuda:0".
        alpha:    blend weight. alpha ~ 0.6 means we trust warped previous
                  colors a bit more than the fresh prediction.
        flow_mode:"dummy" (now) or "raft" (later, on GPU).
        debug:    if True, returns extra tensors in the debug dict.

    Returns:
        ab_fused: torch tensor [1,2,H,W] on `device`
        dbg:      dict with optional debug info
    """
    # Move chroma to device
    ab_prev_t = _to_ab_tensor(ab_prev, device)
    ab_pred_t = _to_ab_tensor(ab_pred, device)

    # Compute flow from prev → curr (currently dummy / zero field)
    flow = compute_flow(prev_rgb, curr_rgb, device=device, mode=flow_mode)
    # flow: [2,H,W] on device
    warped_prev = warp_with_flow(ab_prev_t.squeeze(0), flow)  # [2,H,W]
    warped_prev = warped_prev.unsqueeze(0)                     # [1,2,H,W]

    # Blend warped previous chroma and new prediction
    alpha_t = float(alpha)
    ab_fused = alpha_t * warped_prev + (1.0 - alpha_t) * ab_pred_t

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["flow"] = flow.detach().cpu()
        dbg["warped_prev"] = warped_prev.detach().cpu()
        dbg["ab_pred"] = ab_pred_t.detach().cpu()
        dbg["ab_fused"] = ab_fused.detach().cpu()

    return ab_fused, dbg