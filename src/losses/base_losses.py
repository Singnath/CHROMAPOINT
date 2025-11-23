import torch
import torch.nn.functional as F

def l1_ab(ab_pred, ab_gt):
    return F.l1_loss(ab_pred, ab_gt)

def tv_smooth(ab_pred, weight=0.0):
    if weight <= 0: return torch.tensor(0.0, device=ab_pred.device)
    dx = (ab_pred[:,:, :,1:] - ab_pred[:,:,:, :-1]).abs().mean()
    dy = (ab_pred[:,:,1:,:] - ab_pred[:,:, :-1,:]).abs().mean()
    return weight*(dx+dy)

def sat_penalty(ab_pred, thresh=110.0, weight=0.0):
    if weight <= 0: return torch.tensor(0.0, device=ab_pred.device)
    mag = torch.linalg.norm(ab_pred, dim=1)  # BxHxW
    excess = (mag - thresh).clamp_min(0)
    return weight*excess.mean()