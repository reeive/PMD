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
    
    PRM (Prototype Representation Memory) 的核心组件:
    - topo score: centrality from patient hypergraph (拓扑中心性) - eigenvector centrality via power iteration
    - agree score: neighbor agreement in class embedding space (邻域一致性)
    - conf score: class confidence / availability (置信度)
    - gate mechanism: **patient-level** gate g_n ∈ {0,1}，只选择 top-q% representative patients
      用于更新 prototype，抑制 outliers，降低 semantic overwriting
    
    论文核心设计:
    - Gate 是 patient-level (g_n)，而非 class-level (g_{n,c})
    - 一个患者要么所有区域都参与更新，要么都不参与
    - Gate 基于 topology-aware centrality score 选择
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
                 # Gate 参数
                 gate_ratio: float = 0.5,         # 只保留 top-q% 的患者参与原型更新
                 gate_min_samples: int = 2,       # 最少保留的样本数
                 gate_score_thresh: float = 0.0,  # 可选: 绝对分数阈值 (0=不使用)
                 patient_level_gate: bool = True): # 是否使用 patient-level gate (论文要求)
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.weighted_H = weighted_H
        self.a_topo = a_topo
        self.b_agree = b_agree
        self.d_conf = d_conf
        self.kappa = kappa
        self.mix_uniform = mix_uniform
        # Gate 参数
        self.gate_ratio = gate_ratio
        self.gate_min_samples = gate_min_samples
        self.gate_score_thresh = gate_score_thresh
        self.patient_level_gate = patient_level_gate  # 论文要求 patient-level

    @torch.no_grad()
    def forward(self,
                z_patient: torch.Tensor,      # [B, Dp] - patient embedding (来自 encoder features)
                f_cls: torch.Tensor,          # [B, C, D]   (C=3 for WT/TC/ET)
                conf_cls: torch.Tensor,       # [B, C]      in [0,1]
                return_gate_mask: bool = False):  # 是否返回 gate mask
        """
        计算 reliability-weighted omega 和 patient-level gate
        
        Args:
            z_patient: [B, Dp] 患者级 embedding (应来自 hypergraph-fused encoder features E4/E5)
            f_cls: [B, C, D] 区域级特征 (WT/TC/ET proposals)
            conf_cls: [B, C] 区域置信度
            return_gate_mask: 是否返回 gate mask
            
        Returns:
            omega: [B, C] reliability 权重，每列和为 1
            gate_mask: [B, C] gate 掩码 (patient-level 展开到 class)
        """
        B, C, D = f_cls.shape
        device = z_patient.device

        # ========== 1) 构建 patient hypergraph ==========
        # 使用 L2 距离在 L2-normalized embeddings 上 (等价于 cosine 距离)
        dist = pairwise_l2(F.normalize(z_patient, dim=1))
        knn_idx, knn_dist = knn_indices(dist, k=self.k)
        H = build_incidence_from_knn(knn_idx, knn_dist, sigma=self.sigma, weighted=self.weighted_H)
        G, DV, _ = generate_G_from_H(H)

        # ========== 2) Topology-aware centrality score ==========
        # Eigenvector centrality via power iteration
        s_topo = power_iteration_centrality(G, iters=10)  # [B], sum=1; paper: T=10
        # Z-score normalization (per-class 一致性)
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
        # 基于图邻居之间 proposal 的 cosine 相似度
        f_norm = F.normalize(f_cls, dim=-1)  # [B,C,D]
        neigh = f_norm[knn_idx]              # [B,k,C,D] - 邻居特征
        anchor = f_norm.unsqueeze(1)         # [B,1,C,D]
        agree = (anchor * neigh).sum(dim=-1) # [B,k,C] - cosine 相似度
        s_agree = agree.mean(dim=1)          # [B,C] - 邻居平均一致性
        # Per-class z-score normalization
        s_agree_n = (s_agree - s_agree.mean(dim=0, keepdim=True)) / (s_agree.std(dim=0, keepdim=True).clamp_min(1e-6))

        # ========== 4) Confidence score ==========
        # 来自模型预测概率 (已 detach)
        s_conf = conf_cls.clamp(0.0, 1.0)
        # Per-class z-score normalization
        s_conf_n = (s_conf - s_conf.mean(dim=0, keepdim=True)) / (s_conf.std(dim=0, keepdim=True).clamp_min(1e-6))

        # ========== 5) 融合分数 -> raw representativeness score ==========
        s = (self.a_topo * s_topo_n.unsqueeze(1)
             + self.b_agree * s_agree_n
             + self.d_conf * s_conf_n)  # [B,C]

        # ========== 6) Hypergraph Gate: Patient-level gate ==========
        # 论文核心: g_n ∈ {0,1} 是 patient-level，一个患者所有区域共享同一 gate 状态
        gate_mask = torch.ones(B, C, dtype=torch.bool, device=device)
        
        if self.gate_ratio < 1.0 and B > self.gate_min_samples:
            if self.patient_level_gate:
                # ===== Patient-level gate (论文设计) =====
                # 使用所有 class 的平均分数作为 patient-level 代表性分数
                s_patient = s.mean(dim=1)  # [B]
                
                # 计算保留的患者数
                n_keep = max(self.gate_min_samples, int(B * self.gate_ratio))
                n_keep = min(n_keep, B)
                
                # 选择 top-q 高分患者
                topk_vals, topk_idx = s_patient.topk(n_keep, dim=0, largest=True, sorted=False)
                
                # 可选: 绝对分数阈值过滤
                if self.gate_score_thresh > 0:
                    valid_mask = topk_vals >= self.gate_score_thresh
                    topk_idx = topk_idx[valid_mask]
                
                # 构建 patient-level gate
                gate_patient = torch.zeros(B, dtype=torch.bool, device=device)
                if topk_idx.numel() > 0:
                    gate_patient[topk_idx] = True
                else:
                    gate_patient[s_patient.argmax()] = True
                
                # 展开到所有 class (patient-level gate 对所有区域一致)
                gate_mask = gate_patient.unsqueeze(1).expand(-1, C)  # [B, C]
            else:
                # ===== Class-level gate (备选) =====
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
        
        # ========== 7) 计算 omega (在 gated set 上 softmax) ==========
        # 将未通过 gate 的样本分数设为 -inf (softmax 后权重接近 0)
        s_gated = s.clone()
        s_gated[~gate_mask] = float('-inf')
        
        # Softmax 在患者维度 (dim=0) 上进行，每个 class 独立归一化
        # 结果: Σ_n ω_{n,c} = 1 for each c
        omega = torch.softmax(self.kappa * s_gated, dim=0)  # [B, C]
        
        # 处理全 -inf 的情况 (理论上不会发生)
        omega = torch.where(torch.isnan(omega) | torch.isinf(omega), 
                           torch.zeros_like(omega), omega)
        
        # 确保每个类别权重和为 1
        omega_sum = omega.sum(dim=0, keepdim=True).clamp_min(1e-8)
        omega = omega / omega_sum

        # ========== 8) 可选: 混合 uniform 避免少数样本主导 ==========
        if self.mix_uniform > 0:
            uni = gate_mask.float() / gate_mask.float().sum(dim=0, keepdim=True).clamp_min(1)
            omega = (1 - self.mix_uniform) * omega + self.mix_uniform * uni
            omega = omega / omega.sum(dim=0, keepdim=True).clamp_min(1e-8)
        
        if return_gate_mask:
            return omega, gate_mask
        return omega  # [B,C]
    
    def get_gate_stats(self, gate_mask: torch.Tensor) -> dict:
        """返回 gate 统计信息，用于日志"""
        B, C = gate_mask.shape
        return {
            "total_samples": B,
            "gated_per_class": gate_mask.sum(dim=0).tolist(),
            "gate_ratio_actual": (gate_mask.float().mean(dim=0)).tolist(),
        }
