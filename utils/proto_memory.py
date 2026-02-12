# utils/proto_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeMemoryTopoEMA(nn.Module):
    """
    Maintain prototypes P_c and update by topo-weighted EMA:
    P_c <- (1-eta)*P_c + eta * sum_i omega_{i,c} f_{i,c}
    """
    def __init__(self, num_classes: int, feat_dim: int, eta: float = 0.01, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.eta = eta
        P = torch.zeros(num_classes, feat_dim)
        if device is not None:
            P = P.to(device)
        self.register_buffer("prototypes", P)  # [C,D]
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool, device=P.device))

    @torch.no_grad()
    def update(self, f_cls: torch.Tensor, omega: torch.Tensor, eta: float = None, avail: torch.Tensor = None):
        """
        f_cls: [B,C,D]
        omega: [B,C] (sum across B = 1 per class)
        avail: [B,C] optional gate in {0,1} or [0,1] to suppress invalid regions (e.g. ET absent)
        """
        if eta is None:
            eta = self.eta
        B, C, D = f_cls.shape
        assert C == self.num_classes and D == self.feat_dim

        if avail is not None:
            omega = omega * avail
            omega = omega / omega.sum(dim=0, keepdim=True).clamp_min(1e-8)

        # weighted mean per class: [C,D]
        mean = torch.einsum("bc,bcd->cd", omega, f_cls)

        support = None
        if avail is not None:
            support = (avail.sum(dim=0) > 1e-6)
        else:
            support = torch.ones(C, dtype=torch.bool, device=f_cls.device)

        for c in range(C):
            if not bool(support[c].item()):
                continue
            if not self.initialized[c]:
                self.prototypes[c] = mean[c]
                self.initialized[c] = True
            else:
                self.prototypes[c] = (1.0 - eta) * self.prototypes[c] + eta * mean[c]

        # normalize only initialized classes
        idx = self.initialized.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            self.prototypes[idx] = F.normalize(self.prototypes[idx], dim=1)
