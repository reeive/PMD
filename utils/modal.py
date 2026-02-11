# utils/modal.py
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F

MODS = ["t1", "t2", "flair", "t1ce"]
SUBDIR = {"t1": "imgs_t1", "t2": "imgs_t2", "flair": "imgs_flair", "t1ce": "imgs_t1ce"}

def _expand_allowed(allowed_mask, B: int, device):
    if allowed_mask is None:
        base = torch.ones(4, device=device, dtype=torch.float32)
    else:
        base = allowed_mask if torch.is_tensor(allowed_mask) else torch.as_tensor(allowed_mask, dtype=torch.float32)
        base = base.to(device=device, dtype=torch.float32)
    if base.dim() == 2 and base.size(0) == B and base.size(1) == 4:
        return base.clamp(0, 1)
    if base.dim() != 1:
        base = base.view(-1)
    base = base.clamp(0, 1).unsqueeze(0).repeat(B, 1)
    return base

def _assemble_4ch_from_ids(
    data_root: str,
    ids_list,
    imgs_1c: torch.Tensor,
    current_mode: str,
    allowed_mask=None,
    rand_missing: bool = True,
    p_drop: float = 0.3,
    device=None,
):
    out_dev = imgs_1c.device if device is None else device
    B, _, H, W = imgs_1c.shape
    x4 = imgs_1c.new_zeros((B, 4, H, W))
    pmask = torch.zeros(B, 4, device=imgs_1c.device, dtype=torch.float32)
    cur_idx = MODS.index(current_mode)
    x4[:, cur_idx:cur_idx + 1] = imgs_1c
    pmask[:, cur_idx] = 1.0
    for k, m in enumerate(MODS):
        if k == cur_idx:
            continue
        vals = []
        pres = []
        for sid in ids_list:
            p = Path(data_root) / SUBDIR[m] / f"{sid}.npy"
            if p.exists():
                arr = np.load(p.as_posix())
                if arr.ndim == 2:
                    arr = arr[None, ...]
                t = torch.from_numpy(arr).float()
                if t.shape[-2:] != (H, W):
                    t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
                vals.append(t)
                pres.append(1.0)
            else:
                vals.append(torch.zeros(1, H, W, dtype=torch.float32))
                pres.append(0.0)
        vals_t = torch.stack(vals, dim=0).to(imgs_1c.device)
        x4[:, k:k + 1] = vals_t
        pmask[:, k] = torch.tensor(pres, device=imgs_1c.device, dtype=torch.float32)
    allowed = _expand_allowed(allowed_mask, B, imgs_1c.device)
    keep = torch.ones_like(pmask)
    if rand_missing:
        base = (pmask > 0) & (allowed > 0)
        prob = torch.full_like(pmask, float(p_drop))
        prob[:, cur_idx] = 0.0
        drop = torch.bernoulli(prob) * base.float()
        keep = (1.0 - drop).clamp(0.0, 1.0)
    must_fix = (keep * pmask * allowed).sum(dim=1) < 0.5
    if must_fix.any():
        keep[must_fix, cur_idx] = 1.0
    eff = pmask * allowed * keep
    x4 = x4 * eff.view(B, 4, 1, 1)
    k_present = eff.sum(dim=1)
    scale = torch.ones_like(k_present)
    mask = k_present > 1.5
    if mask.any():
        scale[mask] = (4.0 / k_present[mask])
    x4 = x4 * scale.view(B, 1, 1, 1)
    return x4.to(device=out_dev, dtype=torch.float32, non_blocking=True), eff.to(device=out_dev)

def _discover_prev_modalities(current_mode: str) -> List[str]:
    rb_dir = Path("replay_buffer")
    if not rb_dir.exists():
        return []
    modes = []
    pat = re.compile(r"^replay_buffer_(.+)\.pth$")
    for f in rb_dir.iterdir():
        m = pat.match(f.name)
        if m:
            mm = m.group(1)
            if mm != current_mode:
                modes.append(mm)
    return sorted(list(dict.fromkeys(modes)))
