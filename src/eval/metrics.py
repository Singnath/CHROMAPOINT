# src/eval/metrics.py
from __future__ import annotations
import numpy as np

# skimage is already in your env from Step 2
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# LPIPS is optional. If not installed, we skip it gracefully.
try:
    import lpips  # pip install lpips
    _lpips_available = True
except Exception:
    lpips = None
    _lpips_available = False

def psnr(img_pred_u8: np.ndarray, img_gt_u8: np.ndarray) -> float:
    """PSNR on uint8 RGB images [H,W,3]."""
    return float(sk_psnr(img_gt_u8, img_pred_u8, data_range=255))

def ssim(img_pred_u8: np.ndarray, img_gt_u8: np.ndarray) -> float:
    """SSIM on uint8 RGB images [H,W,3] with multichannel=True."""
    return float(sk_ssim(img_gt_u8, img_pred_u8, channel_axis=2, data_range=255))

class LPIPSWrapper:
    """Lazy LPIPS model that accepts uint8 and converts to normalized torch tensors internally."""
    def __init__(self, net: str = "alex", device: str = "cpu"):
        if not _lpips_available:
            self.model = None
            self.device = "cpu"
            return
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    def __bool__(self):  # bool(LPIPSWrapper(...))
        return self.model is not None

    def __call__(self, img_pred_u8: np.ndarray, img_gt_u8: np.ndarray) -> float:
        import torch
        if self.model is None:
            raise RuntimeError("LPIPS not available. pip install lpips")
        # to BCHW float in [-1,1]
        t1 = torch.from_numpy(img_pred_u8).permute(2,0,1).unsqueeze(0).float() / 255.0
        t2 = torch.from_numpy(img_gt_u8).permute(2,0,1).unsqueeze(0).float() / 255.0
        t1 = (t1 * 2.0 - 1.0).to(self.device)
        t2 = (t2 * 2.0 - 1.0).to(self.device)
        with torch.no_grad():
            d = self.model(t1, t2).item()
        return float(d)