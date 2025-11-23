import cv2
import numpy as np

def ssim_cut(prev_rgb, curr_rgb, thresh=0.5):
    """Fast luminance similarity proxy; low score => scene cut."""
    prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    curr = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    num = (prev*curr).mean()
    den = (prev**2).mean()**0.5 * (curr**2).mean()**0.5 + 1e-6
    score = num / den
    return score < thresh
#