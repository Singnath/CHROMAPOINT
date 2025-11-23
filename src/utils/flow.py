# src/utils/flow.py
"""
Flow utilities for ChromaPoint.

This file exposes a RAFT-style API but currently uses a **dummy backend** so that
the rest of the temporal pipeline (warp + blend + stabilization) can be developed
and run on your MacBook Air M3 without needing a heavy RAFT install.

Later, on a Windows/NVIDIA GPU machine, you can:
  - clone the official RAFT repo
  - load pretrained RAFT weights
  - replace `compute_flow(..., mode="dummy")` with a true RAFT implementation.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor_rgb(img: ArrayLike, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Convert an HxWx3 uint8 or float32 RGB image to a BCHW float tensor in [0,1].
    """
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB ndarray, got shape {arr.shape}")
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    elif isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 3 and t.shape[0] == 3:
            t = t.unsqueeze(0)  # 1x3xHxW
        elif t.ndim == 3 and t.shape[-1] == 3:
            # HxWx3 -> 1x3xHxW
            t = t.permute(2, 0, 1).unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"Expected BCHW or HWC tensor, got shape {t.shape}")

        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    return t.to(device)


def _dummy_flow(
    prev_rgb: ArrayLike,
    curr_rgb: ArrayLike,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Dummy optical flow: returns zeros with shape [2, H, W].

    This means "no motion" everywhere. It allows the temporal fusion code to run
    without crashing, but does NOT provide real temporal consistency.

    Later, replace this with a real RAFT call on a GPU machine.
    """
    t_prev = _to_tensor_rgb(prev_rgb, device=device)  # 1x3xHxW
    _, _, H, W = t_prev.shape

    flow = torch.zeros(1, 2, H, W, dtype=torch.float32, device=device)
    return flow.squeeze(0)  # 2xHxW


def warp_with_flow(
    img: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    Warp `img` using a 2D flow field.

    Args:
        img:  Tensor of shape [C, H, W] or [1, C, H, W],
              e.g., ab channels, or an RGB frame.
        flow: Tensor of shape [2, H, W] giving (dx, dy) in pixel units.

    Returns:
        Warped image with the same shape as `img`.

    Notes:
        - This uses bilinear sampling (grid_sample) with normalized flow.
        - Positive dx moves content to the right, positive dy moves it down.
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # 1xCxHxW
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)  # 1x2xHxW

    B, C, H, W = img.shape

    # Build base grid in normalized coordinates [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=img.device),
        torch.linspace(-1.0, 1.0, W, device=img.device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # HxWx2
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # BxHxWx2

    # Convert flow in pixels to normalized coords:
    # dx_norm = 2 * dx / (W - 1), dy_norm = 2 * dy / (H - 1)
    dx = flow[:, 0, ...]  # BxHxW
    dy = flow[:, 1, ...]
    dx_norm = 2.0 * dx / max(W - 1, 1)
    dy_norm = 2.0 * dy / max(H - 1, 1)
    flow_norm = torch.stack([dx_norm, dy_norm], dim=-1)  # BxHxWx2

    # New sampling grid = base_grid + flow_norm (note sign: move content by dx,dy)
    grid = base_grid + flow_norm

    # Warp
    warped = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(0)  # CxHxW if B=1


def compute_flow(
    prev_rgb: ArrayLike,
    curr_rgb: ArrayLike,
    device: str | torch.device = "cpu",
    mode: Literal["dummy", "raft"] = "dummy",
) -> torch.Tensor:
    """
    Public API for optical flow.

    Args:
        prev_rgb:  Previous RGB frame as HxWx3 (uint8/float32) or tensor.
        curr_rgb:  Current RGB frame, same shape as prev.
        device:    "cpu", "mps", or CUDA device (e.g., "cuda:0").
        mode:      "dummy" = return zero flow (no motion, for dev on Mac),
                   "raft"  = (to be implemented later on a GPU machine).

    Returns:
        flow: Tensor of shape [2, H, W] in pixel units.

    Usage in temporal fusion pipeline:
        flow = compute_flow(frame_t_minus_1, frame_t, device=device)
        warped_ab = warp_with_flow(prev_ab, flow)
    """
    if mode == "dummy":
        return _dummy_flow(prev_rgb, curr_rgb, device=device)

    elif mode == "raft":
        # TODO: implement RAFT inference here on a GPU machine.
        # Rough outline (for later):
        #   - import RAFT
        #   - load pretrained weights
        #   - prepare inputs (BCHW normalized)
        #   - model(prev, curr) -> flow
        #   - return flow[0] as [2,H,W]
        raise NotImplementedError(
            "RAFT mode not implemented yet. Use mode='dummy' for now "
            "and plug in real RAFT on a GPU later."
        )

    else:
        raise ValueError(f"Unknown flow mode: {mode}")