import torch
import torch.nn as nn
import torch.nn.functional as F

class TACLoss(nn.Module):
    def __init__(self, temperature=0.07, ignore_label=-1, alpha=0.7, smooth=1.0, gamma=1.5):
        super(TACLoss, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.alpha = alpha
        self.smooth = smooth
        self.gamma = gamma

    def global_pool(self, features):
        return torch.mean(features, dim=(2, 3))

    def tversky_similarity(self, f1, f2):
        z1 = f1
        z2 = f2
        z1_exp = z1.unsqueeze(1)
        z2_exp = z2.unsqueeze(0)
        TP = torch.sum(z1_exp * z2_exp, dim=2)
        FP = torch.sum(z1_exp * (1 - z2_exp), dim=2)
        FN = torch.sum((1 - z1_exp) * z2_exp, dim=2)
        sim = (TP + self.smooth) / (TP + self.alpha * FP + (1 - self.alpha) * FN + self.smooth)
        return sim

    def forward(self, features1, features2, labels):
        device = features1.device
        if features1.dim() == 4:
            pooled1 = self.global_pool(features1)
            pooled2 = self.global_pool(features2)
        else:
            pooled1 = features1
            pooled2 = features2

        if labels.dim() == 4 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
        elif labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        labels = labels.view(labels.size(0), -1)
        sample_labels = labels[:, 0]

        sim_matrix = self.tversky_similarity(pooled1, pooled2) / self.temperature
        positive_mask = (sample_labels.unsqueeze(1) == sample_labels.unsqueeze(0)).float()

        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True) + 1e-8
        log_prob = sim_matrix - torch.log(sum_exp)

        positive_count = torch.sum(positive_mask, dim=1)
        valid_mask = positive_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = torch.sum(positive_mask * log_prob, dim=1) / (positive_count + 1e-8)
        loss = - torch.mean(mean_log_prob_pos[valid_mask])
        return loss


_EPS = 1e-6

def _to_prob(x: torch.Tensor, act: str = "sigmoid") -> torch.Tensor:
    if act == "sigmoid":
        return torch.sigmoid(x)
    elif act == "relu6":
        return torch.clamp(F.relu(x), 0, 6) / 6.0
    elif act == "softplus":
        return torch.clamp(F.softplus(x), 0, 10) / 10.0
    else:
        return torch.sigmoid(x)

def pairwise_tversky_sim(A: torch.Tensor, B: torch.Tensor,
                         alpha: torch.Tensor,
                         beta:  torch.Tensor,
                         smooth: float = 1.0,
                         act: str = "sigmoid",
                         drift_aware: bool = False) -> torch.Tensor:
    """
    Pairwise Tversky similarity with optional drift-aware adaptation.
    
    When drift_aware=True:
    
    Args:
        A: instance features [N, D]
        B: prototype features [K, D]
        alpha: FP  (base)
        beta: FN  (base)
    """
    A_ = _to_prob(A, act=act)  # [N,D]
    B_ = _to_prob(B, act=act)  # [K,D]
    A2 = A_.unsqueeze(1)       # [N,1,D]
    B2 = B_.unsqueeze(0)       # [1,K,D]

    TP = (A2 * B2).sum(dim=-1)                # [N,K]
    FN = (B2 * (1.0 - A2)).sum(dim=-1)        # [N,K]
    FP = (A2 * (1.0 - B2)).sum(dim=-1)        # [N,K]

    if drift_aware:
        A_norm = F.normalize(A, dim=-1, eps=_EPS)  # [N, D]
        B_norm = F.normalize(B, dim=-1, eps=_EPS)  # [K, D]
        cos_sim = A_norm @ B_norm.t()
        
        sim_factor = (cos_sim + 1.0) / 2.0         # [N, K]
        
        
        alpha_eff = alpha * sim_factor
        
        beta_eff = 1.0 - sim_factor * (1.0 - beta)
        
        T = (TP + smooth) / (TP + beta_eff * FN + alpha_eff * FP + smooth)
    else:
        T = (TP + smooth) / (TP + beta * FN + alpha * FP + smooth)
    
    return T

def _logit_from_sim(S: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    S = torch.clamp(S, eps, 1.0 - eps)
    return torch.log(S) - torch.log(1.0 - S)

import torch
import torch.nn.functional as F

_EPS = 1e-6

def _mp_nce_from_logits(Lpos: torch.Tensor, Lneg: torch.Tensor = None) -> torch.Tensor:
    """
    Lpos: [N, Kp]
    Lneg: [N, Kn] or None
    """
    if Lneg is None or Lneg.numel() == 0:
        return Lpos.new_tensor(0.0)

    logits = torch.cat([Lpos, Lneg], dim=1)                         # [N, Kp+Kn]
    logits = logits - logits.max(dim=1, keepdim=True).values        
    log_den = torch.logsumexp(logits, dim=1)                        # [N]
    log_pos = torch.logsumexp(logits[:, :Lpos.size(1)], dim=1)      # [N]
    return (log_den - log_pos).mean()

def tv_nce_i2p(inst_feat: torch.Tensor,
               proto_pos: torch.Tensor,
               proto_neg_list: list,
               alpha: torch.Tensor,
               beta:  torch.Tensor,
               tau: float = 0.10,
               smooth: float = 1.0,
               act: str = "sigmoid",
               drift_aware: bool = True) -> torch.Tensor:
    """
    Instance-to-Prototype NCE loss with Tversky similarity.
    
     drift_aware=True  drifted instances ( prototype )
    
    Args:
        inst_feat: instance features [B, D]
        proto_pos: positive prototypes [Kp, D]
        proto_neg_list: list of negative prototypes
        tau: temperature
    """
    S_pos = pairwise_tversky_sim(inst_feat, proto_pos, alpha, beta, smooth, act, 
                                  drift_aware=drift_aware)  # [B,Kp]
    
    S_negs = [pairwise_tversky_sim(inst_feat, n, alpha, beta, smooth, act,
                                    drift_aware=drift_aware)
              for n in proto_neg_list if n is not None]
    S_neg = torch.cat(S_negs, dim=1) if len(S_negs) > 0 else None  # [B,Kn] or None

    tau_val = float(tau) if not torch.is_tensor(tau) else float(tau.item())
    Lpos = _logit_from_sim(S_pos) / max(tau_val, _EPS)
    Lneg = (_logit_from_sim(S_neg) / max(tau_val, _EPS)) if (S_neg is not None) else None
    return _mp_nce_from_logits(Lpos, Lneg)

def tv_nce_p2p(proto_a: torch.Tensor,
               proto_pos: torch.Tensor,
               proto_neg_list: list,
               alpha: torch.Tensor,
               beta:  torch.Tensor,
               tau: float = 0.10,
               smooth: float = 1.0,
               act: str = "sigmoid",
               drift_aware: bool = False) -> torch.Tensor:
    """
    Prototype-to-Prototype NCE loss with Tversky similarity.
    
    """
    S_pos = pairwise_tversky_sim(proto_a, proto_pos, alpha, beta, smooth, act,
                                  drift_aware=drift_aware)    # [Ka,Kp]
    S_negs = [pairwise_tversky_sim(proto_a, n, alpha, beta, smooth, act,
                                    drift_aware=drift_aware)
              for n in proto_neg_list if n is not None]
    S_neg = torch.cat(S_negs, dim=1) if len(S_negs) > 0 else None

    tau_val = float(tau) if not torch.is_tensor(tau) else float(tau.item())
    Lpos = _logit_from_sim(S_pos) / max(tau_val, _EPS)
    Lneg = (_logit_from_sim(S_neg) / max(tau_val, _EPS)) if (S_neg is not None) else None
    return _mp_nce_from_logits(Lpos, Lneg)
