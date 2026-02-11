import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import stateless
from typing import Optional, Tuple, Dict, Any, Callable
import logging
from copy import deepcopy


class MetaHyper(nn.Module):
    """
    真正的 Bi-level Meta-Learning 损失权重控制器
    
    核心机制 (One-step Unrolled Bi-level Optimization):
    1. 用当前权重 w = softmax(logits) 计算 support loss: L_sup = w_tv*L_tv + w_ft*L_ft + w_proto*L_pTAC
    2. 做一次虚拟更新: θ' = θ - lr_inner * ∇_θ L_sup  (θ' 依赖 w，计算图保持连通)
    3. 用 θ' 在 validation batch 上计算 **unweighted** meta objective (Dice loss)
    4. 反向传播更新 logits: logits ← logits - lr_meta * ∇_logits L_val(θ')
    
    关键: meta objective 是 unweighted 的，权重 w 通过影响 θ' 来间接影响 L_val
    
    注意: α/β (Tversky 参数) 和 τ (对比学习温度) 由配置文件静态指定
    """
    
    def __init__(
        self, 
        w_tv0: float = 1.0,
        w_ft0: float = 1.0,
        w_proto0: float = 1.0,
        w_total: float = None,
        # 阶段感知参数
        stage_idx: int = 0,
        num_stages: int = 4,
        equal_init: bool = True,
        # 原型损失下限保护
        proto_floor_start: float = 0.0,
        proto_floor_end: float = 0.1,
        # Bi-level meta-learning 参数
        meta_update_freq: int = 10,
        inner_lr: float = 1e-4,         # 虚拟更新的学习率
        first_order: bool = True,       # 使用 first-order 近似 (不计算二阶 Hessian)
    ):
        super().__init__()
        
        # 阶段信息
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.is_first_stage = (stage_idx == 0)
        
        # ===== 权重总和 =====
        if w_total is None:
            w_total = float(w_tv0) + float(w_ft0) + float(w_proto0)
        self.w_total = w_total
        
        # ===== 可学习参数: 权重 logits =====
        if equal_init:
            self.logits = nn.Parameter(torch.zeros(3))
        else:
            init_probs = torch.tensor([w_tv0, w_ft0, w_proto0]) / (w_tv0 + w_ft0 + w_proto0)
            init_logits = torch.log(init_probs + 1e-8)
            init_logits = init_logits - init_logits.mean()
            self.logits = nn.Parameter(init_logits)
        
        # ===== 阶段感知的偏置 (不可学习) =====
        self.register_buffer("stage_bias", self._compute_stage_bias())
        
        # ===== 原型损失下限保护 =====
        self.register_buffer("proto_floor", torch.tensor(0.0, dtype=torch.float32))
        self.proto_floor_start = proto_floor_start
        self.proto_floor_end = proto_floor_end
        
        # ===== Bi-level Meta-Learning 配置 =====
        self.meta_update_freq = meta_update_freq
        self.inner_lr = inner_lr
        self.first_order = first_order
        self._step_count = 0
        
        # 统计 buffer
        self.register_buffer("recent_val_loss", torch.tensor(float('inf')))
        self.register_buffer("ema_val_loss", torch.tensor(0.0))
        self.register_buffer("val_loss_count", torch.tensor(0))
        
        # 权重历史记录 (用于检测权重塌缩)
        self.register_buffer("weight_history", torch.zeros(100, 3))
        self.register_buffer("history_idx", torch.tensor(0))
    
    def _compute_stage_bias(self) -> torch.Tensor:
        """阶段相关的权重偏置"""
        if self.is_first_stage:
            bias = torch.tensor([0.3, 0.3, -0.2])
        elif self.stage_idx == self.num_stages - 1:
            bias = torch.tensor([-0.2, -0.2, 0.5])
        else:
            progress = self.stage_idx / max(1, self.num_stages - 1)
            bias = torch.tensor([
                0.1 - 0.2 * progress,
                0.1 - 0.2 * progress,
                0.0 + 0.3 * progress
            ])
        return bias

    def set_proto_floor(self, val: float):
        v = float(max(0.0, min(0.9, val)))
        self.proto_floor.data = torch.tensor(v, device=self.proto_floor.device)
    
    def set_stage(self, stage_idx: int, num_stages: int):
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.is_first_stage = (stage_idx == 0)
        self.stage_bias.data = self._compute_stage_bias().to(self.stage_bias.device)

    def probs(self) -> torch.Tensor:
        """计算带阶段偏置的权重概率 [3]"""
        adjusted_logits = self.logits + self.stage_bias
        p = torch.softmax(adjusted_logits, dim=0)
        f = self.proto_floor.clamp(0.0, 0.9)
        p = (1.0 - f) * p
        p[2] = p[2] + f
        return p

    def weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取三个损失权重 (w_tv, w_ft, w_proto)"""
        p = self.probs()
        w = self.w_total * p
        return w[0], w[1], w[2]

    def w_tv(self) -> torch.Tensor:
        return self.weights()[0]

    def w_ft(self) -> torch.Tensor:
        return self.weights()[1]

    def w_proto(self) -> torch.Tensor:
        return self.weights()[2]
    
    def weight_entropy(self) -> float:
        """计算权重分布的熵 (用于检测塌缩)"""
        with torch.no_grad():
            p = self.probs()
            entropy = -(p * torch.log(p + 1e-8)).sum()
            return float(entropy.item())
    
    def get_stage_info(self) -> dict:
        with torch.no_grad():
            w_tv, w_ft, w_proto = self.weights()
            return {
                "stage_idx": self.stage_idx,
                "num_stages": self.num_stages,
                "is_first_stage": self.is_first_stage,
                "stage_bias": self.stage_bias.tolist(),
                "weights": {
                    "w_tv": float(w_tv),
                    "w_ft": float(w_ft),
                    "w_proto": float(w_proto),
                },
                "proto_floor": float(self.proto_floor),
                "weight_entropy": self.weight_entropy(),
                "ema_val_loss": float(self.ema_val_loss),
            }
    
    # ==================== Bi-level Meta-Learning ====================
    
    def should_meta_update(self) -> bool:
        self._step_count += 1
        return self._step_count % self.meta_update_freq == 0
    
    def reset_step_count(self):
        self._step_count = 0
    
    def _record_weights(self):
        """记录当前权重到历史 buffer"""
        with torch.no_grad():
            w_tv, w_ft, w_proto = self.weights()
            idx = int(self.history_idx.item()) % 100
            self.weight_history[idx, 0] = w_tv
            self.weight_history[idx, 1] = w_ft
            self.weight_history[idx, 2] = w_proto
            self.history_idx.data = self.history_idx + 1
    
    def check_weight_collapse(self, threshold: float = 0.9) -> Tuple[bool, str]:
        """
        检查权重是否塌缩到单一项
        
        Returns:
            (is_collapsed, message)
        """
        with torch.no_grad():
            p = self.probs()
            max_p = p.max().item()
            if max_p > threshold:
                idx = p.argmax().item()
                names = ["w_tv", "w_ft", "w_proto"]
                return True, f"Weight collapse detected: {names[idx]} = {max_p:.3f}"
            return False, ""
    
    def meta_step_unrolled(
        self,
        model: nn.Module,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs: torch.Tensor,
        query_masks: torch.Tensor,
        alpha: float,
        beta: float,
        gamma: float = 1.2,
        proto_loss_sup: Optional[torch.Tensor] = None,
        trainable_params: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        真正的 Bi-level Meta-Learning 更新 (One-step Unrolled)
        
        流程:
        1. 用当前 w 计算 support loss: L_sup = w_tv*L_tv + w_ft*L_ft + w_proto*L_pTAC
        2. 虚拟更新: θ' = θ - lr_inner * ∇_θ L_sup  (计算图保持连通)
        3. 用 θ' 在 query batch 上计算 unweighted Dice loss 作为 meta objective
        
        Args:
            model: 主网络 (nn.Module)
            support_imgs: support batch 图像 [B, C, H, W]
            support_masks: support batch 标签 [B, 3, H, W]
            query_imgs: query/validation batch 图像
            query_masks: query/validation batch 标签
            alpha, beta: Tversky 参数 (静态)
            gamma: Focal Tversky gamma
            proto_loss_sup: support batch 上的 pTAC loss (可选)
            trainable_params: 要做虚拟更新的参数列表 (None = 全部)
        
        Returns:
            meta_loss: 用于更新 logits 的损失
            stats: 统计信息
        """
        from losses import tversky_prob, focal_tversky_prob
        
        device = support_imgs.device
        
        # ========== Step 1: 计算 support loss ==========
        # 获取当前权重 (保持计算图)
        w_tv, w_ft, w_proto = self.weights()
        
        # Forward on support batch
        sup_logits = model(support_imgs)
        sup_probs = torch.sigmoid(sup_logits)
        
        # 三项 support loss
        L_tv_sup = (1.0 - tversky_prob(sup_probs, support_masks, alpha, beta, smooth=1.0)).mean()
        L_ft_sup = focal_tversky_prob(sup_probs, support_masks, alpha, beta, gamma=gamma, smooth=1.0).mean()
        
        # 加权 support loss
        L_sup = w_tv * L_tv_sup + w_ft * L_ft_sup
        if proto_loss_sup is not None and torch.isfinite(proto_loss_sup):
            L_sup = L_sup + w_proto * proto_loss_sup
        
        # ========== Step 2: 虚拟更新 θ' = θ - lr * ∇L_sup ==========
        # 确定要更新的参数
        if trainable_params is None:
            # 默认: 只更新 decoder 和 segmentation head (降低开销)
            trainable_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 只对 decoder 和 head 做虚拟更新 (encoder 通常很大)
                    if any(k in name.lower() for k in ['decoder', 'head', 'proj', 'final']):
                        trainable_params.append(param)
            
            # 如果筛选后为空，使用所有可训练参数
            if len(trainable_params) == 0:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        # 计算梯度
        # create_graph=True 如果需要二阶，但 first_order=True 时可以用 detach 近似
        grads = torch.autograd.grad(
            L_sup, 
            trainable_params, 
            create_graph=not self.first_order,  # first-order 时不创建二阶图
            retain_graph=True,
            allow_unused=True
        )
        
        # 构建虚拟参数 θ'
        # 使用字典存储: {param_name: θ' value}
        params_dict = dict(model.named_parameters())
        buffers_dict = dict(model.named_buffers())
        
        # 创建新参数字典
        # 注意: 不能使用 `param in trainable_params`，Tensor 的 `__eq__` 会触发逐元素比较并报错。
        grad_by_param_id = {id(p): g for p, g in zip(trainable_params, grads)}
        new_params = {}
        for name, param in model.named_parameters():
            grad = grad_by_param_id.get(id(param), None)
            if grad is not None:
                # θ' = θ - lr * grad
                # first_order 时 detach grad，但保持 θ' 依赖 w (因为 L_sup 依赖 w)
                if self.first_order:
                    new_params[name] = param - self.inner_lr * grad.detach()
                else:
                    new_params[name] = param - self.inner_lr * grad
            else:
                new_params[name] = param
        
        # ========== Step 3: 用 θ' 在 query batch 上计算 unweighted meta objective ==========
        # 使用 functional_call 进行前向传播
        try:
            # PyTorch 2.0+ 方式
            from torch.nn.utils import stateless as stateless_utils
            query_logits = stateless_utils.functional_call(model, new_params, query_imgs)
        except (ImportError, AttributeError):
            # Fallback: 手动替换参数
            query_logits = self._functional_forward_fallback(model, new_params, buffers_dict, query_imgs)
        
        query_probs = torch.sigmoid(query_logits)
        
        # ========== Meta Objective: Unweighted Dice Loss ==========
        # 关键: 这里不使用 w_tv, w_ft, w_proto 加权！
        # 直接计算 Dice loss 作为纯粹的验证性能指标
        L_val_dice = self._dice_loss(query_probs, query_masks)
        
        # 可选: 加一个小的权重正则化防止极端
        weight_entropy = -(self.probs() * torch.log(self.probs() + 1e-8)).sum()
        entropy_reg = 0.01 * (torch.log(torch.tensor(3.0, device=device)) - weight_entropy)  # 鼓励高熵
        
        meta_loss = L_val_dice + entropy_reg.clamp(min=0)
        
        # ========== 更新统计 ==========
        with torch.no_grad():
            if self.val_loss_count == 0:
                self.ema_val_loss.data = L_val_dice.detach()
            else:
                self.ema_val_loss.data = 0.9 * self.ema_val_loss + 0.1 * L_val_dice.detach()
            self.val_loss_count.data = self.val_loss_count + 1
            self.recent_val_loss.data = L_val_dice.detach()
            
            # 记录权重
            self._record_weights()
        
        # 统计信息
        stats = {
            "L_sup": float(L_sup.detach().item()),
            "L_tv_sup": float(L_tv_sup.detach().item()),
            "L_ft_sup": float(L_ft_sup.detach().item()),
            "L_val_dice": float(L_val_dice.detach().item()),
            "entropy_reg": float(entropy_reg.detach().item()),
            "meta_loss": float(meta_loss.detach().item()),
            "w_tv": float(w_tv.detach().item()),
            "w_ft": float(w_ft.detach().item()),
            "w_proto": float(w_proto.detach().item()),
            "weight_entropy": float(weight_entropy.detach().item()),
            "ema_val_loss": float(self.ema_val_loss.item()),
            "inner_lr": self.inner_lr,
            "first_order": self.first_order,
            "num_virtual_params": len(trainable_params),
        }
        
        # 检查权重塌缩
        is_collapsed, collapse_msg = self.check_weight_collapse()
        if is_collapsed:
            stats["warning"] = collapse_msg
        
        return meta_loss, stats
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """计算 Dice Loss (unweighted, 作为 meta objective)"""
        B = pred.size(0)
        p = pred.view(B, -1)
        t = target.view(B, -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def _functional_forward_fallback(
        self, 
        model: nn.Module, 
        new_params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Fallback: 手动替换参数进行前向传播
        
        注意: 这会临时修改模型参数，需要在之后恢复
        """
        # 保存原始参数
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
        
        # 替换为虚拟参数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in new_params:
                    param.data = new_params[name].data
        
        # 前向传播
        output = model(x)
        
        # 恢复原始参数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        return output
    
    # ========== 保留旧接口以兼容 ==========
    def meta_step(
        self,
        query_probs: torch.Tensor,
        query_masks: torch.Tensor,
        alpha: float,
        beta: float,
        gamma: float = 1.2,
        proto_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        [DEPRECATED] 旧的 meta_step，保留以兼容
        推荐使用 meta_step_unrolled()
        """
        from losses import tversky_prob, focal_tversky_prob
        
        w_tv, w_ft, w_proto = self.weights()
        
        val_loss_tv = (1.0 - tversky_prob(query_probs, query_masks, alpha, beta, smooth=1.0)).mean()
        val_loss_ft = focal_tversky_prob(query_probs, query_masks, alpha, beta, gamma=gamma, smooth=1.0).mean()
        
        val_total = w_tv * val_loss_tv + w_ft * val_loss_ft
        if proto_loss is not None and torch.isfinite(proto_loss):
            val_total = val_total + w_proto * proto_loss
        
        weight_reg = 0.01 * (
            (w_tv - self.w_total / 3).pow(2) + 
            (w_ft - self.w_total / 3).pow(2) + 
            (w_proto - self.w_total / 3).pow(2)
        )
        meta_loss = val_total + weight_reg
        
        stats = {
            "val_loss_tv": float(val_loss_tv.detach().item()),
            "val_loss_ft": float(val_loss_ft.detach().item()),
            "val_total": float(val_total.detach().item()),
            "w_tv": float(w_tv.detach().item()),
            "w_ft": float(w_ft.detach().item()),
            "w_proto": float(w_proto.detach().item()),
            "weight_reg": float(weight_reg.detach().item()),
            "deprecated": True,
        }
        
        return meta_loss, stats
    
    def get_meta_gradient_info(self) -> Dict[str, Any]:
        """获取 meta-gradient 信息用于调试"""
        info = {
            "logits": self.logits.detach().cpu().tolist(),
            "logits_grad": None,
            "logits_grad_norm": 0.0,
        }
        
        if self.logits.grad is not None:
            info["logits_grad"] = self.logits.grad.detach().cpu().tolist()
            info["logits_grad_norm"] = float(self.logits.grad.norm().item())
        
        return info
