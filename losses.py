import torch
import torch.nn.functional as F

epsilon = 1e-5
smooth = 1.0

def dsc(y_pred, y_true, smooth=1.0):

    y_pred_f = y_pred.view(y_pred.size(0), -1)
    y_true_f = y_true.view(y_true.size(0), -1)
    intersection = (y_pred_f * y_true_f).sum(dim=1)
    score = (2. * intersection + smooth) / (y_pred_f.sum(dim=1) + y_true_f.sum(dim=1) + smooth)
    return score.mean()

def dice_loss(y_pred, y_true, smooth=1.0):

    return 1 - dsc(y_pred, y_true, smooth)

def bce_dice_loss(y_pred, y_true, smooth=1.0):

    bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
    d_loss = dice_loss(y_pred, y_true, smooth)
    return bce + d_loss

def confusion_metrics(y_pred, y_true, smooth=1.0):

    y_pred = torch.clamp(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred
    y_true_neg = 1 - y_true

    tp = (y_pred * y_true).sum(dim=[1,2,3])
    fp = (y_pred * y_true_neg).sum(dim=[1,2,3])
    fn = (y_pred_neg * y_true).sum(dim=[1,2,3])

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return precision.mean(), recall.mean()

def true_positive(y_pred, y_true, smooth=1.0):

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_true_pos = torch.round(torch.clamp(y_true, 0, 1))
    tp = (y_pred_pos * y_true_pos).sum(dim=[1,2,3])
    return ((tp + smooth) / (y_true_pos.sum(dim=[1,2,3]) + smooth)).mean()

def true_negative(y_pred, y_true, smooth=1.0):

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_true_neg = 1 - torch.round(torch.clamp(y_true, 0, 1))
    tn = (y_pred_neg * y_true_neg).sum(dim=[1,2,3])
    return ((tn + smooth) / (y_true_neg.sum(dim=[1,2,3]) + smooth)).mean()


def tversky(y_pred, y_true, alpha=0.7, smooth=1.0):

    if y_pred.size(0) != y_true.size(0):
        return torch.tensor(1e-8, device=y_pred.device, dtype=y_pred.dtype)

    y_pred_f = y_pred.view(y_pred.size(0), -1)
    y_true_f = y_true.view(y_true.size(0), -1)
    true_pos = (y_pred_f * y_true_f).sum(dim=1)
    false_neg = (y_true_f * (1 - y_pred_f)).sum(dim=1)
    false_pos = (y_pred_f * (1 - y_true_f)).sum(dim=1)
    tversky_index = (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )
    return tversky_index.mean()

def tversky_loss(y_pred, y_true, alpha=0.7, smooth=1.0):

    return 1 - tversky(y_pred, y_true, alpha, smooth)

def focal_tversky(y_pred, y_true, alpha=0.7, gamma=1.5, smooth=1.0):

    tversky_index = tversky(y_pred, y_true, alpha, smooth)
    return torch.pow((1 - tversky_index), gamma)

def info_nce_i2p(inst_feat, proto_pos, proto_neg_list, tau=0.07):
    q = F.normalize(inst_feat, dim=-1)                  # [B,D]
    pos = F.normalize(proto_pos, dim=-1)                # [K,D]
    logits_pos = q @ pos.t() / tau                      # [B,K]
    logits_negs = []
    for neg in proto_neg_list:
        n = F.normalize(neg, dim=-1)                    # [K,D]
        logits_negs.append(q @ n.t() / tau)
    logits_neg = torch.cat(logits_negs, dim=1) if len(logits_negs)>0 else None
    if logits_neg is None:
        loss = -torch.log_softmax(logits_pos, dim=1).mean()
    else:
        logits = torch.cat([logits_pos, logits_neg], dim=1)   # [B, K + sumK]
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
    return loss

def info_nce_p2p(proto_a, proto_pos, proto_neg_list, tau=0.07):
    a = F.normalize(proto_a, dim=-1)                    # [K,D]
    p = F.normalize(proto_pos, dim=-1)                  # [K,D]
    logits_pos = a @ p.t() / tau                        # [K,K]
    logits_negs = []
    for neg in proto_neg_list:
        n = F.normalize(neg, dim=-1)
        logits_negs.append(a @ n.t() / tau)
    logits_neg = torch.cat(logits_negs, dim=1) if len(logits_negs)>0 else None
    if logits_neg is None:
        loss = -torch.log_softmax(logits_pos, dim=1).mean()
    else:
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
        loss = F.cross_entropy(logits, labels)
    return loss

def tversky_prob(y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 alpha: torch.Tensor,   # α: FP weight
                 beta:  torch.Tensor,   # β: FN weight
                 smooth: float = 1.0) -> torch.Tensor:
    B = y_pred.size(0)
    p = y_pred.view(B, -1)
    q = y_true.view(B, -1)
    tp = (p * q).sum(dim=1)
    fn = (q * (1.0 - p)).sum(dim=1)
    fp = (p * (1.0 - q)).sum(dim=1)
    ti = (tp + smooth) / (tp + beta * fn + alpha * fp + smooth)
    return ti.mean()

def focal_tversky_prob(y_pred: torch.Tensor,
                       y_true: torch.Tensor,
                       alpha: torch.Tensor,   
                       beta:  torch.Tensor,
                       gamma: float = 1.5,
                       smooth: float = 1.0) -> torch.Tensor:
    ti = tversky_prob(y_pred, y_true, alpha, beta, smooth)
    return torch.pow(1.0 - ti, gamma)
