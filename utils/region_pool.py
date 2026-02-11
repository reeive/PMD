# utils/region_pool.py
import torch
import torch.nn.functional as F

def region_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    feat: [B, D, H, W]
    mask: [B, 1, H, W] in [0,1] (soft mask)
    return: [B, D]
    """
    w = mask
    num = (feat * w).sum(dim=(2, 3))
    den = w.sum(dim=(2, 3)).clamp_min(eps)
    return num / den

def region_conf(prob: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    prob: [B,1,H,W] or [B,H,W], predicted prob for that region
    mask: [B,1,H,W] (same region mask; can reuse prob as mask)
    return: [B], confidence in [0,1]
    """
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    num = (prob * mask).sum(dim=(2, 3))
    den = mask.sum(dim=(2, 3)).clamp_min(eps)
    return (num / den).squeeze(1)  # [B]
