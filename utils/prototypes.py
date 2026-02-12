import torch
import torch.nn as nn
import torch.nn.functional as F

REGIONS = ["WT", "TC", "ET"]
MODES = ["t1", "t2", "flair", "t1ce"]

def _l2n(x, eps=1e-6):
    return F.normalize(x, dim=-1, eps=eps)

class PrototypeMemory(nn.Module):
    """
    
    - P_tumor: [4 modalities, 3 regions (WT/TC/ET), 1 prototype, D dims]
    - P_struct: [3 regions, 1 prototype, D dims]
    
    
    """
    def __init__(self, d: int, Kt: int = 1, Ks: int = 1, ema_m: float = 0.01, 
                 learnable: bool = False, device: str = "cuda", sim: str = "cos"):
        """
        Args:
            Kt: tumor prototype slots (paper: 1, one per class)
            Ks: struct prototype slots (paper: 1)
            ema_m: EMA momentum for new value (paper: η=0.99 keep rate → ema_m=1-η=0.01)
                   Update: P = (1-ema_m)*P_old + ema_m*P_new
            learnable: whether prototypes are learnable parameters
            device: device
            sim: similarity metric ("cos" for cosine)
        """
        super().__init__()
        self.d = d
        self.Kt = 1
        self.Ks = 1
        self.m = float(ema_m)
        self.sim = sim
        
        self.register_parameter("P_tumor_main", nn.Parameter(
            torch.zeros(4, 3, self.Kt, d, device=device), requires_grad=learnable))
        self.register_parameter("P_struct_main", nn.Parameter(
            torch.zeros(3, self.Ks, d, device=device), requires_grad=learnable))
        self.register_parameter("P_tumor_noce", nn.Parameter(
            torch.zeros(4, 3, self.Kt, d, device=device), requires_grad=learnable))
        self.register_parameter("P_struct_noce", nn.Parameter(
            torch.zeros(3, self.Ks, d, device=device), requires_grad=learnable))
        
        self.register_buffer("U_tumor_main", torch.zeros(4, 3, self.Kt, device=device))
        self.register_buffer("U_struct_main", torch.zeros(3, self.Ks, device=device))
        self.register_buffer("U_tumor_noce", torch.zeros(4, 3, self.Kt, device=device))
        self.register_buffer("U_struct_noce", torch.zeros(3, self.Ks, device=device))
        
        nn.init.normal_(self.P_tumor_main.data, std=1e-4)
        nn.init.normal_(self.P_struct_main.data, std=1e-4)
        nn.init.normal_(self.P_tumor_noce.data, std=1e-4)
        nn.init.normal_(self.P_struct_noce.data, std=1e-4)

    def _bank_tumor(self, cond: str):
        return self.P_tumor_noce if cond == "no_t1ce" else self.P_tumor_main

    def _bank_struct(self, cond: str):
        return self.P_struct_noce if cond == "no_t1ce" else self.P_struct_main

    def get_tumor(self, m_idx: int, r_idx: int, cond: str = "main") -> torch.Tensor:
        """ (modality, region)  tumor prototype [Kt=1, D]"""
        return (self.P_tumor_noce if cond == "no_t1ce" else self.P_tumor_main)[m_idx, r_idx]

    def get_struct(self, r_idx: int, cond: str = "main") -> torch.Tensor:
        """ region  struct prototype [Ks=1, D]"""
        return (self.P_struct_noce if cond == "no_t1ce" else self.P_struct_main)[r_idx]

    def _sim_mat(self, z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        if self.sim == "cos":
            return _l2n(z) @ _l2n(P).t()
        return _l2n(z) @ _l2n(P).t()

    @torch.no_grad()
    def ema_update_tumor(self, m_idx: int, r_idx: int, z: torch.Tensor, 
                         assign: str = "hard", temp: float = 0.07, cond: str = "main"):
        """
         (modality, region)  tumor prototype  EMA
        
         Kt=1 prototype slot
        
        : P = (1 - m) * P + m * z
        """
        if z is None or z.numel() == 0:
            return
        
        if z.dim() == 2 and z.size(0) > 1:
            z = z.mean(dim=0, keepdim=True)
        if z.dim() == 1:
            z = z.unsqueeze(0)  # [1, D]
        
        P = self._bank_tumor(cond).data[m_idx, r_idx]  # [Kt=1, D]
        
        P[0] = (1.0 - self.m) * P[0] + self.m * z.squeeze(0)
        
        if cond == "no_t1ce":
            self.U_tumor_noce[m_idx, r_idx, 0] += 1.0
        else:
            self.U_tumor_main[m_idx, r_idx, 0] += 1.0

    @torch.no_grad()
    def ema_update_struct(self, r_idx: int, z: torch.Tensor, 
                          assign: str = "hard", temp: float = 0.07, cond: str = "main"):
        """
        
         Ks=1 prototype slot
        """
        if z is None or z.numel() == 0:
            return
        
        if z.dim() == 2 and z.size(0) > 1:
            z = z.mean(dim=0, keepdim=True)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        P = self._bank_struct(cond).data[r_idx]  # [Ks=1, D]
        
        P[0] = (1.0 - self.m) * P[0] + self.m * z.squeeze(0)
        
        if cond == "no_t1ce":
            self.U_struct_noce[r_idx, 0] += 1.0
        else:
            self.U_struct_main[r_idx, 0] += 1.0

    def align_loss(self, reduction: str = "mean") -> torch.Tensor:
        """
        
         Kt=Ks=1squeeze  slot  L2
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
