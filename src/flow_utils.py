import numpy as np
import cv2
import torch
import torch.nn.functional as F

def warp_ab(ab_prev, flow):
    """Warp previous frame's ab (prev->curr) using bilinear sampling."""
    h, w = ab_prev.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x2 = xx + flow[...,0]; y2 = yy + flow[...,1]
    gx = (x2 / max(w-1,1)) * 2 - 1
    gy = (y2 / max(h-1,1)) * 2 - 1
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1)).float()[None]       # 1xHxWx2
    ab_t = torch.from_numpy(ab_prev).permute(2,0,1).unsqueeze(0)             # 1x2xHxW
    ab_w = F.grid_sample(ab_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return ab_w[0].permute(1,2,0).numpy()

def edge_softmask(rgb_uint8):
    """Downweight alpha near strong edges to avoid ghosting."""
    g = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = mag / (mag.max() + 1e-6)
    return 0.5 + 0.5*(1.0 - mag)  # [0.5..1]

def flow_confidence_from_fb(fwd, bwd):
    """Forward–backward consistency -> confidence [0..1]."""
    h,w,_ = fwd.shape
    yy,xx = np.mgrid[0:h,0:w]
    x2 = xx + fwd[...,0]; y2 = yy + fwd[...,1]
    bwd_warp = cv2.remap(bwd, x2.astype(np.float32), y2.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(fwd + bwd_warp, axis=-1)
    return np.exp(-err / 2.0)

class FlowEstimator:
    """
    CPU-friendly Farnebäck (works on M3). RAFT can be added later for GPU.
    """
    def __init__(self, device="auto", method="farneback"):
        if device == "auto":
            if torch.backends.mps.is_available():  self.device = "mps"
            elif torch.cuda.is_available():        self.device = "cuda"
            else:                                   self.device = "cpu"
        else:
            self.device = device
        self.method = "farneback"

    def compute_flow(self, rgb_prev, rgb_curr):
        prev = cv2.cvtColor(rgb_prev, cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2GRAY)
        fwd = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        bwd = cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return fwd.astype(np.float32), bwd.astype(np.float32)
#