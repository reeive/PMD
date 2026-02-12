# utils/patient_graph_weight.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, D]
    return: [B, B] squared l2 distance
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x2 = (x * x).sum(dim=1, keepdim=True)  # [B,1]
    dist = x2 + x2.t() - 2.0 * (x @ x.t())
    return dist.clamp_min_(0.0)

def knn_indices(dist: torch.Tensor, k: int):
    """
    dist: [B,B], smaller is closer
    return: knn_idx [B,k], knn_dist [B,k]
    """
    B = dist.size(0)
    k = min(k, B - 1)
    # exclude self by setting diagonal to +inf
    dist2 = dist.clone()
    dist2.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = torch.topk(dist2, k=k, dim=1, largest=False, sorted=True)
    return knn_idx, knn_dist

def build_incidence_from_knn(knn_idx: torch.Tensor, knn_dist: torch.Tensor,
                             sigma: float = 1.0, weighted: bool = True):
    """
    Build hypergraph incidence matrix H for "each node creates one hyperedge".
    Hyperedge e_i = {i} U KNN(i)
    H: [B, E=B] with entries in {0,1} or weights
    """
    B, k = knn_idx.shape
    device = knn_idx.device
    H = torch.zeros(B, B, device=device)  # [node, hyperedge]
    # always connect self to its hyperedge
    H[torch.arange(B, device=device), torch.arange(B, device=device)] = 1.0

    if k > 0:
        if weighted:
            # weight = exp(-d / (sigma * mean_d))  (more stable than d^2 for small B)
            mean_d = knn_dist.mean().clamp_min(1e-6)
            w = torch.exp(-knn_dist / (sigma * mean_d))
        else:
            w = torch.ones_like(knn_dist)

        # for each hyperedge e_i, add neighbors as incident nodes
        # H[neighbor, i] = w
        row = knn_idx.reshape(-1)                         # neighbors
        col = torch.arange(B, device=device).repeat_interleave(k)  # hyperedge id i
        H[row, col] = w.reshape(-1)
    return H

def generate_G_from_H(H: torch.Tensor, eps: float = 1e-8):
    """
    H: [B, E] where E=B
    Return normalized G: [B,B], and node degree DV
    """
    B, E = H.shape
    W = torch.ones(E, device=H.device)  # hyperedge weight = 1
    DV = H.sum(dim=1).clamp_min(eps)    # [B]
    DE = H.sum(dim=0).clamp_min(eps)    # [E]

    DV2 = torch.diag(DV.pow(-0.5))
    invDE = torch.diag(DE.pow(-1.0))
    W = torch.diag(W)
    G = DV2 @ H @ W @ invDE @ H.t() @ DV2
    return G, DV, DE

def power_iteration_centrality(G: torch.Tensor, iters: int = 20):
    """
    Eigenvector-like centrality on G (non-negative).
    """
    B = G.size(0)
    v = torch.ones(B, device=G.device) / B
    for _ in range(iters):
        v = G @ v
        v = v / (v.sum().clamp_min(1e-8))
    return v  # [B], sums to 1

class PatientGraphWeighter(nn.Module):
    """
    Output omega_{p,c} for prototype update with hypergraph-based gating.
    
    PRM (Prototype Representation Memory) :
    - topo score: centrality from patient hypergraph () - eigenvector centrality via power iteration
    - agree score: neighbor agreement in class embedding space ()
    - conf score: class confidence / availability ()
    
    - Gate  patient-level (g_n) class-level (g_{n,c})
    """
    def __init__(self,
                 k: int = 5,
                 sigma: float = 1.0,
                 weighted_H: bool = True,
                 a_topo: float = 0.5,
                 b_agree: float = 0.3,
                 d_conf: float = 0.2,
                 kappa: float = 5.0,
                 mix_uniform: float = 0.1,
                 gate_ratio: float = 0.5,
                 gate_min_samples: int = 2,
                 gate_score_thresh: float = 0.0,
                 patient_level_gate: bool = True):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.weighted_H = weighted_H
        self.a_topo = a_topo
        self.b_agree = b_agree
        self.d_conf = d_conf
        self.kappa = kappa
        self.mix_uniform = mix_uniform
        self.gate_ratio = gate_ratio
        self.gate_min_samples = gate_min_samples
        self.gate_score_thresh = gate_score_thresh
        self.patient_level_gate = patient_level_gate

    @torch.no_grad()
    def forward(self,
                z_patient: torch.Tensor,
                f_cls: torch.Tensor,          # [B, C, D]   (C=3 for WT/TC/ET)
                conf_cls: torch.Tensor,       # [B, C]      in [0,1]
                return_gate_mask: bool = False):
        """
        
        Args:
            z_patient: [B, Dp]  embedding ( hypergraph-fused encoder features E4/E5)
            f_cls: [B, C, D]  (WT/TC/ET proposals)
            conf_cls: [B, C]
            
        Returns:
            omega: [B, C] reliability  1
            gate_mask: [B, C] gate  (patient-level  class)
        """
        B, C, D = f_cls.shape
        device = z_patient.device

        dist = pairwise_l2(F.normalize(z_patient, dim=1))
        knn_idx, knn_dist = knn_indices(dist, k=self.k)
        H = build_incidence_from_knn(knn_idx, knn_dist, sigma=self.sigma, weighted=self.weighted_H)
        G, DV, _ = generate_G_from_H(H)

        # ========== 2) Topology-aware centrality score ==========
        # Eigenvector centrality via power iteration
        s_topo = power_iteration_centrality(G, iters=10)  # [B], sum=1; paper: T=10
        s_topo_n = (s_topo - s_topo.mean()) / (s_topo.std().clamp_min(1e-6))

        # neighbors list for agreement
        k = knn_idx.size(1)
        if k == 0:
            # degenerate: all weights uniform
            omega = torch.full((B, C), 1.0 / B, device=device)
            gate_mask = torch.ones(B, C, dtype=torch.bool, device=device)
            if return_gate_mask:
                return omega, gate_mask
            return omega

        # ========== 3) Neighbor-consistency score ==========
        f_norm = F.normalize(f_cls, dim=-1)  # [B,C,D]
        neigh = f_norm[knn_idx]
        anchor = f_norm.unsqueeze(1)         # [B,1,C,D]
        agree = (anchor * neigh).sum(dim=-1)
        s_agree = agree.mean(dim=1)
        # Per-class z-score normalization
        s_agree_n = (s_agree - s_agree.mean(dim=0, keepdim=True)) / (s_agree.std(dim=0, keepdim=True).clamp_min(1e-6))

        # ========== 4) Confidence score ==========
        s_conf = conf_cls.clamp(0.0, 1.0)
        # Per-class z-score normalization
        s_conf_n = (s_conf - s_conf.mean(dim=0, keepdim=True)) / (s_conf.std(dim=0, keepdim=True).clamp_min(1e-6))

        s = (self.a_topo * s_topo_n.unsqueeze(1)
             + self.b_agree * s_agree_n
             + self.d_conf * s_conf_n)  # [B,C]

        # ========== 6) Hypergraph Gate: Patient-level gate ==========
        gate_mask = torch.ones(B, C, dtype=torch.bool, device=device)
        
        if self.gate_ratio < 1.0 and B > self.gate_min_samples:
            if self.patient_level_gate:
                s_patient = s.mean(dim=1)  # [B]
                
                n_keep = max(self.gate_min_samples, int(B * self.gate_ratio))
                n_keep = min(n_keep, B)
                
                topk_vals, topk_idx = s_patient.topk(n_keep, dim=0, largest=True, sorted=False)
                
                if self.gate_score_thresh > 0:
                    valid_mask = topk_vals >= self.gate_score_thresh
                    topk_idx = topk_idx[valid_mask]
                
                gate_patient = torch.zeros(B, dtype=torch.bool, device=device)
                if topk_idx.numel() > 0:
                    gate_patient[topk_idx] = True
                else:
                    gate_patient[s_patient.argmax()] = True
                
                gate_mask = gate_patient.unsqueeze(1).expand(-1, C)  # [B, C]
            else:
                for c in range(C):
                    s_c = s[:, c]
                    n_keep = max(self.gate_min_samples, int(B * self.gate_ratio))
                    n_keep = min(n_keep, B)
                    topk_vals, topk_idx = s_c.topk(n_keep, dim=0, largest=True, sorted=False)
                    
                    if self.gate_score_thresh > 0:
                        valid_mask = topk_vals >= self.gate_score_thresh
                        topk_idx = topk_idx[valid_mask]
                    
                    gate_c = torch.zeros(B, dtype=torch.bool, device=device)
                    if topk_idx.numel() > 0:
                        gate_c[topk_idx] = True
                    else:
                        gate_c[s_c.argmax()] = True
                    gate_mask[:, c] = gate_c
        
        s_gated = s.clone()
        s_gated[~gate_mask] = float('-inf')
        
        omega = torch.softmax(self.kappa * s_gated, dim=0)  # [B, C]
        
        omega = torch.where(torch.isnan(omega) | torch.isinf(omega), 
                           torch.zeros_like(omega), omega)
        
        omega_sum = omega.sum(dim=0, keepdim=True).clamp_min(1e-8)
        omega = omega / omega_sum

        if self.mix_uniform > 0:
            uni = gate_mask.float() / gate_mask.float().sum(dim=0, keepdim=True).clamp_min(1)
            omega = (1 - self.mix_uniform) * omega + self.mix_uniform * uni
            omega = omega / omega.sum(dim=0, keepdim=True).clamp_min(1e-8)
        
        if return_gate_mask:
            return omega, gate_mask
        return omega  # [B,C]
    
    def get_gate_stats(self, gate_mask: torch.Tensor) -> dict:
        B, C = gate_mask.shape
        return {
            "total_samples": B,
            "gated_per_class": gate_mask.sum(dim=0).tolist(),
            "gate_ratio_actual": (gate_mask.float().mean(dim=0)).tolist(),
        }
