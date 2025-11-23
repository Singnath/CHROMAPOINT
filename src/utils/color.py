import numpy as np
from skimage import color as ski_color
import torch

def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    return ski_color.rgb2lab(rgb_uint8.astype(np.float32)/255.0)

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = lab.copy()
    lab[...,0] = np.clip(lab[...,0], 0.0, 100.0)
    lab[...,1:] = np.clip(lab[...,1:], -110.0, 110.0)
    rgb = ski_color.lab2rgb(lab)
    return (np.clip(rgb, 0, 1)*255.0).astype(np.uint8)

def L_from_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(rgb_uint8)
    L = lab[...,0] * (255.0/100.0)
    return L.astype(np.uint8)

def compose_Lab(L_uint8: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """L_uint8 HxW, ab HxWx2 float -> Lab HxWx3 float (L in [0..100])."""
    L = L_uint8.astype(np.float32) * (100.0/255.0)
    return np.stack([L, ab[...,0], ab[...,1]], axis=-1)

def to_tensor_L(L_uint8: np.ndarray) -> torch.Tensor:
    L = torch.from_numpy(L_uint8.astype(np.float32)/255.0)  # [0..1]
    L = L*2.0 - 1.0
    return L.unsqueeze(0)  # 1xHxW

def to_tensor_ab(ab: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((ab / 110.0).astype(np.float32)).permute(2,0,1)

def from_tensor_ab(ab_t: torch.Tensor) -> np.ndarray:
    """
    Accepts:
      - [2,H,W]  -> returns [H,W,2]
      - [B,2,H,W] -> returns [B,H,W,2]
    Values are in [-1,1] approx; rescale to native ab by *110.
    """
    t = ab_t.detach().cpu()
    if t.dim() == 3:
        ab = t.permute(1, 2, 0).numpy() * 110.0
        return ab.astype(np.float32)
    elif t.dim() == 4:
        ab = t.permute(0, 2, 3, 1).numpy() * 110.0
        return ab.astype(np.float32)
    else:
        raise ValueError(f"from_tensor_ab expects [2,H,W] or [B,2,H,W], got {tuple(t.shape)}")

def denorm_L(L_t: torch.Tensor) -> np.ndarray:
    """
    Accepts:
      - [1,H,W]  -> returns [H,W] uint8
      - [B,1,H,W] -> returns [B,H,W] uint8
    Inverse of [-1,1] -> [0..255].
    """
    t = L_t.detach().cpu()
    if t.dim() == 3:
        L = ((t.squeeze(0).numpy() + 1.0)/2.0) * 255.0
        return L.clip(0,255).astype(np.uint8)
    elif t.dim() == 4:
        L = ((t + 1.0) / 2.0) * 255.0
        L = L.squeeze(1).numpy()
        return L.clip(0,255).astype(np.uint8)
    else:
        raise ValueError(f"denorm_L expects [1,H,W] or [B,1,H,W], got {tuple(t.shape)}")