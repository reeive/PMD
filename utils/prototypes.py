import torch
import torch.nn as nn
import torch.nn.functional as F

REGIONS = ["WT", "TC", "ET"]
MODES = ["t1", "t2", "flair", "t1ce"]

def _l2n(x, eps=1e-6):
    return F.normalize(x, dim=-1, eps=eps)

class PrototypeMemory(nn.Module):
    """
    Prototype Memory: 维护 class-wise global prototypes
    
    论文设计: "one prototype per subregion/class"
    - P_tumor: [4 modalities, 3 regions (WT/TC/ET), 1 prototype, D dims]
    - P_struct: [3 regions, 1 prototype, D dims]
    
    每个 prototype 在 feature space 中总结 "globally shared semantics"，
    作为跨模态/跨阶段的 persistent reference（语义锚点）。
    
    更新方式: weighted EMA，由 PRM 的 hypergraph gate 筛选后的
    representative samples 进行加权更新，抑制 outliers。
    """
    def __init__(self, d: int, Kt: int = 1, Ks: int = 1, ema_m: float = 0.05, 
                 learnable: bool = False, device: str = "cuda", sim: str = "cos"):
        """
        Args:
            d: prototype 维度
            Kt: tumor prototype slots 数量（论文要求=1，即 one per class）
            Ks: struct prototype slots 数量（论文要求=1）
            ema_m: EMA momentum (new_value 的权重)
            learnable: 是否将 prototype 作为可学习参数
            device: 设备
            sim: 相似度计算方式 ("cos" for cosine)
        """
        super().__init__()
        # 强制 Kt=Ks=1 以符合论文 "one prototype per class" 的设计
        self.d = d
        self.Kt = 1  # 每个 (modality, region) 一个 prototype
        self.Ks = 1  # 每个 struct region 一个 prototype
        self.m = float(ema_m)
        self.sim = sim
        
        # Prototype 张量: [4 modalities, 3 regions, 1 slot, D dims]
        # 使用 register_parameter 确保进入 state_dict
        self.register_parameter("P_tumor_main", nn.Parameter(
            torch.zeros(4, 3, self.Kt, d, device=device), requires_grad=learnable))
        self.register_parameter("P_struct_main", nn.Parameter(
            torch.zeros(3, self.Ks, d, device=device), requires_grad=learnable))
        self.register_parameter("P_tumor_noce", nn.Parameter(
            torch.zeros(4, 3, self.Kt, d, device=device), requires_grad=learnable))
        self.register_parameter("P_struct_noce", nn.Parameter(
            torch.zeros(3, self.Ks, d, device=device), requires_grad=learnable))
        
        # 更新计数器 (用于统计)
        self.register_buffer("U_tumor_main", torch.zeros(4, 3, self.Kt, device=device))
        self.register_buffer("U_struct_main", torch.zeros(3, self.Ks, device=device))
        self.register_buffer("U_tumor_noce", torch.zeros(4, 3, self.Kt, device=device))
        self.register_buffer("U_struct_noce", torch.zeros(3, self.Ks, device=device))
        
        # 初始化: 小随机值防止全零
        nn.init.normal_(self.P_tumor_main.data, std=1e-4)
        nn.init.normal_(self.P_struct_main.data, std=1e-4)
        nn.init.normal_(self.P_tumor_noce.data, std=1e-4)
        nn.init.normal_(self.P_struct_noce.data, std=1e-4)

    def _bank_tumor(self, cond: str):
        return self.P_tumor_noce if cond == "no_t1ce" else self.P_tumor_main

    def _bank_struct(self, cond: str):
        return self.P_struct_noce if cond == "no_t1ce" else self.P_struct_main

    def get_tumor(self, m_idx: int, r_idx: int, cond: str = "main") -> torch.Tensor:
        """获取指定 (modality, region) 的 tumor prototype [Kt=1, D]"""
        return (self.P_tumor_noce if cond == "no_t1ce" else self.P_tumor_main)[m_idx, r_idx]

    def get_struct(self, r_idx: int, cond: str = "main") -> torch.Tensor:
        """获取指定 region 的 struct prototype [Ks=1, D]"""
        return (self.P_struct_noce if cond == "no_t1ce" else self.P_struct_main)[r_idx]

    def _sim_mat(self, z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        if self.sim == "cos":
            return _l2n(z) @ _l2n(P).t()
        return _l2n(z) @ _l2n(P).t()

    @torch.no_grad()
    def ema_update_tumor(self, m_idx: int, r_idx: int, z: torch.Tensor, 
                         assign: str = "hard", temp: float = 0.07, cond: str = "main"):
        """
        对指定 (modality, region) 的 tumor prototype 进行 EMA 更新
        
        由于 Kt=1（单 prototype），直接更新唯一的 slot。
        z 应该是由 PRM gate 筛选后的 weighted mean。
        
        更新公式: P = (1 - m) * P + m * z
        """
        if z is None or z.numel() == 0:
            return
        
        # 归一化输入形状
        if z.dim() == 2 and z.size(0) > 1:
            z = z.mean(dim=0, keepdim=True)  # 多个样本取平均
        if z.dim() == 1:
            z = z.unsqueeze(0)  # [1, D]
        
        P = self._bank_tumor(cond).data[m_idx, r_idx]  # [Kt=1, D]
        
        # 单 prototype 模式：直接更新唯一的 slot (k=0)
        # 不再需要 argmax 选择
        P[0] = (1.0 - self.m) * P[0] + self.m * z.squeeze(0)
        
        # 更新计数
        if cond == "no_t1ce":
            self.U_tumor_noce[m_idx, r_idx, 0] += 1.0
        else:
            self.U_tumor_main[m_idx, r_idx, 0] += 1.0

    @torch.no_grad()
    def ema_update_struct(self, r_idx: int, z: torch.Tensor, 
                          assign: str = "hard", temp: float = 0.07, cond: str = "main"):
        """
        对指定 region 的 struct prototype 进行 EMA 更新
        
        由于 Ks=1（单 prototype），直接更新唯一的 slot。
        """
        if z is None or z.numel() == 0:
            return
        
        # 归一化输入形状
        if z.dim() == 2 and z.size(0) > 1:
            z = z.mean(dim=0, keepdim=True)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        P = self._bank_struct(cond).data[r_idx]  # [Ks=1, D]
        
        # 单 prototype 模式：直接更新唯一的 slot (k=0)
        P[0] = (1.0 - self.m) * P[0] + self.m * z.squeeze(0)
        
        # 更新计数
        if cond == "no_t1ce":
            self.U_struct_noce[r_idx, 0] += 1.0
        else:
            self.U_struct_main[r_idx, 0] += 1.0

    def align_loss(self, reduction: str = "mean") -> torch.Tensor:
        """
        计算 main 和 no_t1ce 条件下 prototype 的对齐损失
        
        由于 Kt=Ks=1，squeeze 掉 slot 维度后直接计算 L2 距离
        """
        # [4, 3, 1, D] -> [4, 3, D] (squeeze slot dim)
        Pt_m = _l2n(self.P_tumor_main).squeeze(2)
        Pt_c = _l2n(self.P_tumor_noce).squeeze(2)
        # [3, 1, D] -> [3, D]
        Ps_m = _l2n(self.P_struct_main).squeeze(1)
        Ps_c = _l2n(self.P_struct_noce).squeeze(1)
        
        lt = (Pt_m - Pt_c).pow(2).sum(dim=-1)  # [4, 3]
        ls = (Ps_m - Ps_c).pow(2).sum(dim=-1)  # [3]
        
        if reduction == "mean":
            return (lt.mean() + ls.mean())
        if reduction == "sum":
            return (lt.sum() + ls.sum())
        return torch.cat([lt.flatten(), ls.flatten()], dim=0).mean()
