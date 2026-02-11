from typing import List, Sequence, Optional, Tuple, Dict
import random
import torch
import torch.nn as nn

def _to_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().tolist()
    return [x]

def tversky_sim(p: torch.Tensor, q: torch.Tensor, alpha: float = 0.7, beta: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    tp = (p * q).sum(dim=(2, 3))
    fp = ((1.0 - q) * p).sum(dim=(2, 3))
    fn = ((1.0 - p) * q).sum(dim=(2, 3))
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)

class BalanceQueue:
    def __init__(self, max_size: int, channels: int, height: int, width: int, dtype: torch.dtype = torch.float16, device: str = "cpu"):
        self.max_size = int(max_size)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.dtype = dtype
        self.device = device
        self._probs = torch.empty((0, self.channels, self.height, self.width), dtype=dtype, device=device)
        self._patients: List[str] = []
        self._modalities: List[str] = []
        self._presence_bits: List[int] = []

    @property
    def size(self) -> int:
        return self._probs.shape[0]

    def enqueue(self, probs: torch.Tensor, patient_ids: Sequence, modality_ids: Sequence, presence_bits: Optional[Sequence[int]] = None):
        if probs.numel() == 0:
            return
        assert probs.dim() == 4 and probs.shape[1:] == (self.channels, self.height, self.width)
        pt = _to_list(patient_ids)
        md = _to_list(modality_ids)
        assert len(pt) == probs.shape[0] and len(md) == probs.shape[0]
        if presence_bits is None:
            pb = [0] * probs.shape[0]
        else:
            pb = [int(x) for x in _to_list(presence_bits)]
            assert len(pb) == probs.shape[0]
        probs_cpu = probs.detach().to(self.device, dtype=self.dtype)
        self._probs = torch.cat([self._probs, probs_cpu], dim=0)
        self._patients.extend([str(x) for x in pt])
        self._modalities.extend([str(x) for x in md])
        self._presence_bits.extend(pb)
        if self.size > self.max_size:
            overflow = self.size - self.max_size
            self._probs = self._probs[overflow:].contiguous()
            self._patients = self._patients[overflow:]
            self._modalities = self._modalities[overflow:]
            self._presence_bits = self._presence_bits[overflow:]

    def _eligible_indices(self, forbid_modalities: Optional[Sequence[str]] = None, forbid_patients: Optional[Sequence[str]] = None) -> List[int]:
        fm = set([str(x) for x in _to_list(forbid_modalities)])
        fp = set([str(x) for x in _to_list(forbid_patients)])
        idx = []
        for i, (pid, mod) in enumerate(zip(self._patients, self._modalities)):
            if (mod in fm) or (pid in fp):
                continue
            idx.append(i)
        return idx

    def sample_balanced(self, num: int, forbid_modalities: Optional[Sequence[str]] = None, forbid_patients: Optional[Sequence[str]] = None, prefer_presence_bits: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
        cand = self._eligible_indices(forbid_modalities, forbid_patients)
        if not cand:
            return torch.empty(0, self.channels, self.height, self.width, dtype=self.dtype, device=self.device), []
        by_mod: Dict[str, List[int]] = {}
        for i in cand:
            by_mod.setdefault(self._modalities[i], []).append(i)
        for k in by_mod.keys():
            random.shuffle(by_mod[k])
        mods = list(by_mod.keys())
        S = min(int(num), len(cand))
        if S <= 0:
            return torch.empty(0, self.channels, self.height, self.width, dtype=self.dtype, device=self.device), []
        pick: List[int] = []
        if prefer_presence_bits is not None:
            same = [i for i in cand if self._presence_bits[i] == int(prefer_presence_bits)]
            other = [i for i in cand if self._presence_bits[i] != int(prefer_presence_bits)]
            random.shuffle(same)
            random.shuffle(other)
            need = S
            take = min(len(same), int(max(1, 0.7 * S)))
            pick.extend(same[:take])
            remain = need - len(pick)
            if remain > 0:
                pick.extend(other[:remain])
        else:
            cur = {m: 0 for m in mods}
            while len(pick) < S:
                for m in mods:
                    if cur[m] < len(by_mod[m]):
                        pick.append(by_mod[m][cur[m]])
                        cur[m] += 1
                        if len(pick) >= S:
                            break
        probs_cpu = self._probs[pick]
        return probs_cpu, pick

def build_teacher_queue(prev_model: nn.Module, images: torch.Tensor, patient_ids: Sequence, modality_ids: Sequence, max_size: int, batch_size: int = 64, device: str = "cuda", out_device: str = "cpu", dtype: torch.dtype = torch.float16, in_channels: int = 1) -> BalanceQueue:
    assert images.ndim == 4
    N = images.shape[0]
    if in_channels > 1 and images.shape[1] == 1:
        images = images.repeat(1, in_channels, 1, 1)
    prev_model.eval().to(device)
    q: Optional[BalanceQueue] = None
    with torch.no_grad():
        for i in range(0, N, batch_size):
            sl = slice(i, min(i + batch_size, N))
            x = images[sl].to(device, non_blocking=True)
            y = torch.sigmoid(prev_model(x)).detach()
            if q is None:
                _, C_pred, H_pred, W_pred = y.shape
                q = BalanceQueue(max_size=int(max_size), channels=int(C_pred), height=int(H_pred), width=int(W_pred), dtype=dtype, device=out_device)
            mods = [str(m) for m in modality_ids[sl]]
            bits = []
            for m in mods:
                idx = {"t1": 0, "t2": 1, "flair": 2, "t1ce": 3}.get(m, -1)
                b = 0 if idx < 0 else (1 << idx)
                bits.append(b)
            q.enqueue(y, patient_ids=patient_ids[sl], modality_ids=mods, presence_bits=bits)
    if q is None:
        _, C_pred, H_pred, W_pred = 1, 1, images.shape[2], images.shape[3]
        q = BalanceQueue(max_size=int(max_size), channels=int(C_pred), height=int(H_pred), width=int(W_pred), dtype=dtype, device=out_device)
    return q

class TACWithQueues(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 1.5, tau: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.last_stats = None

    def _info_nce_channels(self, p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        S = p_probs.shape[0]
        sim_pos = tversky_sim(p_probs, q_probs, alpha=self.alpha, beta=self.beta)
        q_neg1 = torch.roll(q_probs, shifts=1, dims=1)
        sim_neg1 = tversky_sim(p_probs, q_neg1, alpha=self.alpha, beta=self.beta)
        if S > 1:
            perm = torch.randperm(S, device=p_probs.device)
            q_neg2 = q_probs[perm]
            sim_neg2 = tversky_sim(p_probs, q_neg2, alpha=self.alpha, beta=self.beta)
            sim_neg = torch.cat([sim_neg1, sim_neg2], dim=1)
        else:
            sim_neg = sim_neg1
        logit_pos = sim_pos / self.tau
        logit_neg = sim_neg / self.tau
        max_pos = logit_pos.max(dim=1, keepdim=True).values
        max_neg = logit_neg.max(dim=1, keepdim=True).values
        Z_pos = (logit_pos - max_pos).exp().mean(dim=1)
        Z_neg = (logit_neg - max_neg).exp().mean(dim=1)
        loss = -torch.log(Z_pos / (Z_pos + Z_neg + 1e-12)).mean()
        return loss

    def forward(self, cur_probs: torch.Tensor, cur_patient_ids: Sequence[str], cur_modality_ids: Sequence[str], teacher_queue: BalanceQueue, current_queue: Optional[BalanceQueue] = None, cur_presence_bits: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = cur_probs.device
        B, C, H, W = cur_probs.shape
        batch_mods = [str(x) for x in _to_list(cur_modality_ids)]
        assert len(set(batch_mods)) == 1
        cur_mod = batch_mods[0]
        batch_pats = [str(x) for x in _to_list(cur_patient_ids)]
        prefer_bits = None
        if cur_presence_bits is not None and cur_presence_bits.numel() > 0:
            vals, cnt = torch.unique(cur_presence_bits.to(torch.int64), return_counts=True)
            prefer_bits = int(vals[cnt.argmax()].item())
        S_A = min(B, teacher_queue.size)
        loss_A = cur_probs.new_tensor(0.0)
        hit_A = 0
        if S_A > 0:
            qA_cpu, idxA = teacher_queue.sample_balanced(num=S_A, forbid_modalities=[cur_mod], forbid_patients=batch_pats, prefer_presence_bits=prefer_bits)
            S_A = qA_cpu.shape[0]
            if S_A > 0:
                qA = qA_cpu.to(device=device, dtype=cur_probs.dtype, non_blocking=True)
                sel = torch.randperm(B, device=device)[:S_A]
                pA = cur_probs[sel]
                loss_A = self._info_nce_channels(pA, qA)
                hit_A = S_A
        loss_B = cur_probs.new_tensor(0.0)
        hit_B = 0
        if current_queue is not None and current_queue.size > 0 and teacher_queue.size > 0:
            S_B = min(current_queue.size, teacher_queue.size, B)
            if S_B > 0:
                d_cpu, didx = current_queue.sample_balanced(num=S_B, forbid_modalities=[], forbid_patients=batch_pats, prefer_presence_bits=prefer_bits)
                r_cpu, ridx = teacher_queue.sample_balanced(num=d_cpu.shape[0], forbid_modalities=[cur_mod], forbid_patients=[], prefer_presence_bits=prefer_bits)
                S_B = min(d_cpu.shape[0], r_cpu.shape[0])
                if S_B > 0:
                    pB = d_cpu[:S_B].to(device=device, dtype=cur_probs.dtype, non_blocking=True)
                    qB = r_cpu[:S_B].to(device=device, dtype=cur_probs.dtype, non_blocking=True)
                    loss_B = self._info_nce_channels(pB, qB)
                    hit_B = S_B
        parts = []
        if hit_A > 0:
            parts.append(loss_A)
        if hit_B > 0:
            parts.append(loss_B)
        loss = torch.stack(parts).mean() if parts else cur_probs.new_tensor(0.0)
        self.last_stats = {"num_anchor": int(hit_A + hit_B), "num_pos_found": int(hit_A + hit_B)}
        return loss
