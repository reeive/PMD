# -*- coding: utf-8 -*-
import math, copy, torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -------------------------------
# Utils
# -------------------------------

def _flatten_embed(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 4:
        return z.mean(dim=(2, 3))
    return z


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def conditional_distribution(z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    s = pairwise_cosine(z, z) / tau
    s = s - torch.diag(torch.diag(s))
    p = F.softmax(s, dim=-1)
    return p


def kl_rowwise(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = (p + 1e-8).clamp_max(1.0)
    q = (q + 1e-8).clamp_max(1.0)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


# -------------------------------
# (A) Prototype Bank with optional spatial augmentation
# -------------------------------
@dataclass
class ProtoCfg:
    dim: int
    momentum: float = 0.5
    aug_enable: bool = False
    aug_k: int = 4
    aug_noise: float = 0.01
    device: str = "cuda"


class PrototypeBank:
    def __init__(self, num_classes: int, cfg: ProtoCfg):
        self.num_classes = num_classes
        self.cfg = cfg
        self.register = torch.zeros(num_classes, cfg.dim, device=cfg.device)
        self.count = torch.zeros(num_classes, device=cfg.device)

    @torch.no_grad()
    def _ema_update(self, c: int, v: torch.Tensor):
        if self.count[c] == 0:
            self.register[c] = v
        else:
            self.register[c] = self.cfg.momentum * self.register[c] + (1 - self.cfg.momentum) * v
        self.count[c] += 1

    @torch.no_grad()
    def update_from_batch(self, z: torch.Tensor, y: torch.Tensor):
        z = _flatten_embed(z)
        for cls in y.unique().tolist():
            m = (y == cls)
            if m.any():
                v = z[m].mean(dim=0)
                self._ema_update(int(cls), v)

    @torch.no_grad()
    def get(self, classes: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if classes is None:
            classes = [i for i in range(self.num_classes) if self.count[i] > 0]
        if len(classes) == 0:
            return torch.empty(0, self.cfg.dim, device=self.cfg.device), torch.empty(0, dtype=torch.long, device=self.cfg.device)
        protos = self.register[classes]
        labels = torch.tensor(classes, device=self.cfg.device, dtype=torch.long)
        return protos, labels

    @torch.no_grad()
    def sample_augmented(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        protos, labels = self.get()
        if protos.numel() == 0:
            return [protos], labels
        if not self.cfg.aug_enable:
            return [protos], labels
        outs = []
        for _ in range(self.cfg.aug_k):
            v = protos
            noise = torch.randn_like(v) * self.cfg.aug_noise
            v = v + noise
            outs.append(v)
        return outs, labels


# -------------------------------
# (B) Multi-Teacher KD (PAMK-style PD losses)
# -------------------------------
@dataclass
class MTCfg:
    tau: float = 0.1
    eta: float = 10.0
    device: str = "cuda"


class MultiTeacherKD:
    def __init__(self, student: torch.nn.Module, old_teacher: Optional[torch.nn.Module], new_teacher: Optional[torch.nn.Module], cfg: MTCfg,
                 feature_module_name: Optional[str] = None):
        self.student = student
        self.old_teacher = old_teacher
        self.new_teacher = new_teacher
        self.cfg = cfg
        self.feature_module_name = feature_module_name

    @torch.no_grad()
    def _forward_embed(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(model, 'forward_embed'):
            z = model.forward_embed(x)
        else:
            z = model(x)
            if isinstance(z, (list, tuple)):
                z = z[0]
        return _flatten_embed(z)

    def pd_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pN = None
            if self.new_teacher is not None:
                zN = self._forward_embed(self.new_teacher, x)
                pN = conditional_distribution(zN, self.cfg.tau)
            pO = None
            if self.old_teacher is not None:
                zO = self._forward_embed(self.old_teacher, x)
                pO = conditional_distribution(zO, self.cfg.tau)
        zC = self._forward_embed(self.student, x)
        qC = conditional_distribution(zC, self.cfg.tau)
        lossN = torch.zeros((), device=x.device)
        lossO = torch.zeros((), device=x.device)
        if pN is not None:
            lossN = kl_rowwise(pN, qC)
        if pO is not None:
            lossO = kl_rowwise(pO, qC)
        return lossN, lossO


# -------------------------------
# (C) SVC / CE heads on prototypes (feature-space)
# -------------------------------
@dataclass
class SvcCeCfg:
    tau: float = 0.1
    lambda_svc: float = 0.01
    lambda_ce: float = 0.01
    device: str = "cuda"


class SvcCe:
    def __init__(self, projector: Optional[torch.nn.Module], classifier: Optional[torch.nn.Module], cfg: SvcCeCfg):
        self.projector = projector
        self.classifier = classifier
        self.cfg = cfg

    def _proj(self, v: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            return v
        return self.projector(v)

    def svc_loss(self, proto_views: List[torch.Tensor]) -> torch.Tensor:
        if len(proto_views) <= 1:
            return torch.zeros((), device=self.cfg.device)
        qs = [conditional_distribution(self._proj(v), self.cfg.tau) for v in proto_views]
        qbar = torch.stack(qs, dim=0).mean(dim=0).detach()
        ce = 0.
        for q in qs:
            ce = ce + kl_rowwise(qbar, q)
        return ce / len(qs)

    def ce_loss(self, protos: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.classifier is None or protos.numel() == 0:
            return torch.zeros((), device=self.cfg.device)
        logits = self.classifier(self._proj(protos))
        return F.cross_entropy(logits, labels)


# -------------------------------
# Glue: one-call loss builder
# -------------------------------
@dataclass
class PamkWeights:
    lambda_wg: float = 0.01  # reserved if you have extra generator/reg terms
    lambda_pd: float = 0.1
    lambda_svc: float = 0.01
    lambda_ce: float = 0.01


class PAMKLoss:
    def __init__(self, proto_bank: PrototypeBank, kd: MultiTeacherKD, svc_ce: SvcCe, w: PamkWeights):
        self.proto_bank = proto_bank
        self.kd = kd
        self.svc_ce = svc_ce
        self.w = w

    def __call__(self, batch_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        lN, lO = self.kd.pd_loss(batch_x)
        proto_views, plabels = self.proto_bank.sample_augmented()
        l_svc = self.svc_ce.svc_loss(proto_views) * self.w.lambda_svc
        base_protos = proto_views[0] if len(proto_views) > 0 else torch.empty(0, self.proto_bank.cfg.dim, device=batch_x.device)
        l_ce = self.svc_ce.ce_loss(base_protos, plabels) * self.w.lambda_ce
        out = {
            'pamk_pd_N': lN * self.w.lambda_pd,
            'pamk_pd_O': lO * self.w.lambda_pd * 10.0,  # scale old teacher as in paper (eta≈10)
            'pamk_svc': l_svc,
            'pamk_ce': l_ce,
        }
        out['pamk_total'] = sum(out.values())
        return out


# -------------------------------
# Integration helper
# -------------------------------
@dataclass
class PamkInitArgs:
    num_classes: int
    feat_dim: int
    proto_momentum: float = 0.5
    proto_aug_enable: bool = False
    proto_aug_k: int = 4
    proto_aug_noise: float = 0.01
    tau: float = 0.1
    eta: float = 10.0
    lambda_pd: float = 0.1
    lambda_svc: float = 0.01
    lambda_ce: float = 0.01
    device: str = "cuda"


def build_pamk(student: torch.nn.Module, old_teacher: Optional[torch.nn.Module], new_teacher: Optional[torch.nn.Module],
               classifier: Optional[torch.nn.Module], projector: Optional[torch.nn.Module], args: PamkInitArgs) -> Tuple[PAMKLoss, PrototypeBank, MultiTeacherKD, SvcCe]:
    pbank = PrototypeBank(args.num_classes, ProtoCfg(dim=args.feat_dim, momentum=args.proto_momentum, aug_enable=args.proto_aug_enable, aug_k=args.proto_aug_k, aug_noise=args.proto_aug_noise, device=args.device))
    kd = MultiTeacherKD(student, old_teacher, new_teacher, MTCfg(tau=args.tau, eta=args.eta, device=args.device))
    svcce = SvcCe(projector, classifier, SvcCeCfg(tau=args.tau, lambda_svc=args.lambda_svc, lambda_ce=args.lambda_ce, device=args.device))
    pamk = PAMKLoss(pbank, kd, svcce, PamkWeights(lambda_pd=args.lambda_pd, lambda_svc=args.lambda_svc, lambda_ce=args.lambda_ce))
    return pamk, pbank, kd, svcce


# -------------------------------
# Example of minimal training-loop hooks (pseudo)
# -------------------------------
"""
# at stage init
pamk, pbank, kd, svcce = build_pamk(
    student=model,
    old_teacher=old_teacher_model,  # None if first stage
    new_teacher=new_teacher_model,  # optional; pre-warm on new data if available
    classifier=cls_head,            # None if you don't have a classifier; CE loss will be 0
    projector=proj_head,            # None to use identity on embeddings
    args=PamkInitArgs(num_classes=N_C, feat_dim=FEAT_DIM, proto_aug_enable=args.proto_aug, proto_aug_k=args.proto_aug_k,
                      proto_momentum=args.proto_m, tau=args.pamk_tau, eta=args.pamk_eta, lambda_pd=args.pamk_lambda2,
                      lambda_svc=args.pamk_lambda3, lambda_ce=args.pamk_lambda4, device=device)
)

# inside training loop per batch
logits, embeds = model(x)  # ensure embeds is the feature before classifier; or use model.forward_embed
base_loss = criterion(logits, y)
with torch.no_grad():
    pbank.update_from_batch(embeds, y)
extra = pamk(x)
loss = base_loss + extra['pamk_total']
loss.backward()
optim.step()
"""


# -------------------------------
# Segmentation-specific prototype enhancements (pixel-level labels)
# -------------------------------
from typing import Mapping


def _resize_seg_to(feats: torch.Tensor, seg: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
    """Resize integer segmentation map to feats' spatial size using nearest.
    feats: [B,C,Hf,Wf], seg: [B,Hs,Ws] (long) or [B,1,Hs,Ws]
    returns: [B,Hf,Wf] long
    """
    if seg.dim() == 3:
        seg_ = seg.unsqueeze(1).float()
    elif seg.dim() == 4 and seg.size(1) == 1:
        seg_ = seg.float()
    else:
        raise ValueError(f"seg must be [B,H,W] or [B,1,H,W], got {seg.shape}")
    Bh, _, Hf, Wf = feats.shape
    if seg_.shape[-2:] != (Hf, Wf):
        seg_ = F.interpolate(seg_, size=(Hf, Wf), mode=mode)
    return seg_.squeeze(1).long()


class SegPrototypeBank(PrototypeBank):
    """Prototype bank tailored for segmentation (pixel-level labels).
    Adds update_from_segmentation(): per-class masked mean of feature maps with EMA.
    - Supports ignore_index, optional class id mapping, and DDP all-reduce.
    - Works with any embedding tensor shaped [B,C,H,W] (pre-classifier or neck features).
    """
    @torch.no_grad()
    def update_from_segmentation(
        self,
        feats: torch.Tensor,            # [B,C,Hf,Wf]
        seg: torch.Tensor,              # [B,Hs,Ws] or [B,1,Hs,Ws] (long)
        ignore_index: int = 255,
        class_map: Optional[Mapping[int, int]] = None,  # e.g., {1:0, 2:1, 4:2} for WT/TC/ET remap
        resize_mode: str = 'nearest',
        ddp_reduce: bool = False,
    ) -> None:
        assert feats.dim() == 4, f"feats must be [B,C,H,W], got {feats.shape}"
        B, C, Hf, Wf = feats.shape
        segr = _resize_seg_to(feats, seg, mode=resize_mode)
        if class_map is not None:
            segm = segr.clone()
            for src, dst in class_map.items():
                segm[segr == int(src)] = int(dst)
            seg = segm
        else:
            seg = segr

        # classes present (excluding ignore)
        valid_mask = (seg != ignore_index)
        if not valid_mask.any():
            return
        present = torch.unique(seg[valid_mask]).tolist()
        if len(present) == 0:
            return

        # accumulate sums & counts per class (vectorized over classes)
        device = feats.device
        cls_list = [int(c) for c in present]
        sums = torch.zeros(len(cls_list), C, device=device)
        counts = torch.zeros(len(cls_list), device=device)

        # expand once
        feats_f = feats  # [B,C,Hf,Wf]
        for i, c in enumerate(cls_list):
            m = (seg == c)  # [B,Hf,Wf]
            cnt = m.sum()
            counts[i] = cnt
            if cnt > 0:
                mc = m.unsqueeze(1).to(feats_f.dtype)  # [B,1,Hf,Wf]
                sums[i] = (feats_f * mc).sum(dim=(0, 2, 3))

        # optional DDP reduction
        if ddp_reduce:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            except Exception:
                pass

        # finalize EMA updates
        for i, c in enumerate(cls_list):
            cnt = counts[i].item()
            if cnt > 0:
                v = sums[i] / (cnt + 1e-8)
                self._ema_update(int(c), v)

    @torch.no_grad()
    def update_from_masks(
        self,
        feats: torch.Tensor,                    # [B,C,H,W]
        mask_dict: Dict[int, torch.Tensor],     # {class_id: [B,H,W] bool/long}
        ddp_reduce: bool = False,
    ) -> None:
        B, C, Hf, Wf = feats.shape
        device = feats.device
        cls_list = sorted(list(mask_dict.keys()))
        sums = torch.zeros(len(cls_list), C, device=device)
        counts = torch.zeros(len(cls_list), device=device)
        for i, c in enumerate(cls_list):
            m = mask_dict[c]
            if m.dim() == 3:
                m = m
            elif m.dim() == 4 and m.size(1) == 1:
                m = m.squeeze(1)
            else:
                raise ValueError(f"mask for class {c} must be [B,H,W] or [B,1,H,W], got {m.shape}")
            if m.shape[-2:] != (Hf, Wf):
                m = _resize_seg_to(feats, m, mode='nearest')
            m = (m > 0)
            cnt = m.sum()
            counts[i] = cnt
            if cnt > 0:
                mc = m.unsqueeze(1).to(feats.dtype)
                sums[i] = (feats * mc).sum(dim=(0, 2, 3))
        if ddp_reduce:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            except Exception:
                pass
        for i, c in enumerate(cls_list):
            cnt = counts[i].item()
            if cnt > 0:
                v = sums[i] / (cnt + 1e-8)
                self._ema_update(int(c), v)

# Usage (example in training loop):
"""
# feats: encoder feature map [B,C,Hf,Wf]
# seg  : pixel labels [B,Hs,Ws], with ignore_index=255
seg_bank = SegPrototypeBank(num_classes=N_C, cfg=ProtoCfg(dim=FEAT_DIM, momentum=0.5, aug_enable=False, device=device))
# If your dataset uses non-contiguous ids (e.g., BraTS: {0,1,2,4} -> {bg, ET, WT, TC}), provide a map:
class_map = {1: 0, 2: 1, 4: 2}  # example mapping to 3 foreground classes; adjust to your IDs
seg_bank.update_from_segmentation(feats, seg, ignore_index=255, class_map=class_map, resize_mode='nearest', ddp_reduce=True)
# Then you can use seg_bank in PAMKLoss just like PrototypeBank
pamk = PAMKLoss(seg_bank, kd, svcce, PamkWeights(lambda_pd=0.1, lambda_svc=0.01, lambda_ce=0.01))
"""
