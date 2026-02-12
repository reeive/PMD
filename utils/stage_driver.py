# -*- coding: utf-8 -*-
import os
import math
import logging
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from nets.cph import CPH
from losses import tversky_prob, focal_tversky_prob
from dataloader.dataset import BaseDataSets, PatientBatchSampler
from utils.metrics import dice as dice_all
from utils.metrics import batch_dice
from utils.prototypes import PrototypeMemory
from utils.tac_loss import tv_nce_i2p
from utils.per_buffer import select_with_per
from utils.meta_hyper import MetaHyper
from utils.modal import _discover_prev_modalities
from utils.replay import _save_replay, _load_all_replay_buffers, ReplayDataset, MixedBatchSampler
from utils.patient_graph_weight import PatientGraphWeighter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# modality order: t1, t2, flair, t1ce
MODS = ["t1", "t2", "flair", "t1ce"]


@dataclass
class StageConfig:
    # basic paths
    base_dir: str
    data_path: str
    train_list: str
    val_list: str
    img_mode: str

    # fused 4-channel slices (optional; if missing file, fallback to single-modality dirs)
    fused_dir: Optional[str] = "/home/wjz/10.1/BraTS_2019/BraTS_slice/BraTS_fusedslice"
    fused_dtype: str = "float16"

    # legacy mask dir (kept for compatibility)
    mask_dir: Optional[str] = "/home/wjz/10.1/BraTS_2019/BraTS_slice/masks_all"

    # previous stages
    prev_img_modes: Optional[List[str]] = None
    prev_base_dir: Optional[str] = None

    # replay memory
    mem_size: int = 32000
    p_keep: float = 0.10

    # training schedule
    max_epoch: int = 200
    batch_size: int = 128
    images_rate: float = 1.0

    # optimizer & LR
    base_lr: float = 1e-4
    weight_decay: float = 4e-4
    optim_name: str = "adam"
    lr_scheduler: str = "warmupMultistep"  # warmupMultistep | warmupCosine | autoReduce
    step_num_lr: int = 4

    # loss weights (fallback if meta not used / meta fail)
    tversky_w: float = 3.0
    imb_w: float = 3.0
    nce_weight: float = 3.5

    # Tversky / focal parameters
    alpha: float = 0.7      # Tversky α (FP weight, paper: 0.7)
    beta: float = 1.5       # Tversky β (FN weight, paper: 1.5)
    gamma: float = 1.2      # Focal Tversky gamma (paper: 1.2)

    # legacy ablation fields (kept)
    ab_freeze_epochs: int = 15
    ab_kl: float = 1e-3

    # data loader
    in_channels: int = 4
    num_workers_train: int = 8
    num_workers_val: int = 4
    num_workers_meta: int = 4
    prefetch_train: int = 2
    prefetch_val: int = 2
    prefetch_meta: int = 2

    # system
    device: str = "cuda"
    seed: int = 1111
    ddp: bool = False

    # MetaHyper (真正的 Bi-level Meta-Learning 损失权重控制器)
    use_meta: bool = True
    meta_lr: float = 1e-3
    meta_update_freq: int = 10          # Bi-level meta 更新频率 (每 K 步执行一次)
    meta_bilevel: bool = True           # 是否启用 bi-level meta-learning
    meta_unrolled: bool = True          # 是否使用 unrolled bi-level (推荐 True)
    meta_inner_lr: float = 1e-4         # 虚拟更新的学习率
    meta_first_order: bool = True       # 使用 first-order 近似 (推荐 True，降低开销)
    meta_detach: bool = False           # True 表示 meta 只做"控制器输出"，不学习
    meta_freeze_epochs: int = 10        # 前 N 个 epoch 冻结 meta 参数
    meta_equal_init: bool = True        # 三个权重初始相等
    
    # 损失权重初始值 (用于计算 w_total)
    init_lambda_proto: float = 3.5      # pTAC 损失权重
    init_tversky_w: float = 7.0         # Tversky 损失权重
    init_imb_w: float = 8.0             # Focal Tversky 损失权重
    
    # pTAC 对比学习
    tau: float = 0.10                   # pTAC temperature (paper: τ=0.1)
    proto_warmup_epochs: int = 10


    # prototype scheduler
    proto_floor_start: float = 0.05
    proto_floor_end: float = 0.35
    proto_floor_epochs: int = 80
    proto_align_lambda: float = 5e-3
    proto_target_bias: float = 0.0
    proto_gain_start: int = 30
    proto_gain_span: int = 60
    proto_gain_max: float = 2.0
    proto_dim: int = 128

    # ---- EMA teacher (kept for compatibility) ----
    new_teacher_is_ema: bool = True
    new_teacher_ema_m: float = 0.99
    per_use_ema: bool = True
    per_ema_m: float = 0.7

    # replay selection (PER or median)
    per_enable: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_epsilon: float = 1e-6

    # control first stage + fused inputs
    force_first_stage: Optional[bool] = None
    allow_prev_in_4ch: bool = True

    # ---- Replay Training (经验回放) ----
    use_replay: bool = True                   # 是否启用 replay 训练
    replay_ratio: float = 0.3                 # 每个 batch 中 replay 数据的比例
    replay_contrast_weight: float = 1.0       # replay 对比学习损失权重
    replay_stratified: bool = True            # 是否分层抽样 (按模态均衡)

    # ---- Stage-aware Meta (阶段感知) ----
    stage_idx: int = 0                        # 当前阶段索引 (0-based)
    num_stages: int = 4                       # 总阶段数
    meta_equal_init: bool = True              # 三个损失权重是否初始相等
    
    # ---- Bi-level Meta-Learning (元学习) ----
    meta_bilevel: bool = True                 # 是否启用 bi-level meta-learning
    meta_update_freq: int = 10                # 每 N 步进行一次 meta 更新
    meta_query_ratio: float = 0.3             # query batch 占总 batch 的比例
    
    # ---- PRM Gate (原型记忆门控) ----
    prm_gate_ratio: float = 0.5               # 只保留 top-k% 的样本参与原型更新
    prm_gate_min_samples: int = 2             # 最少保留的样本数


def _np_load(path: str, dtype=None):
    arr = np.load(path, allow_pickle=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _patient_from_sid(sid: str) -> str:
    sid = str(sid)
    return sid.rsplit("_", 1)[0]


def _resolve_device(cfg: StageConfig, ddp: bool, local_rank: int) -> torch.device:
    if ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


class FourChFusedDataset(Dataset):
    """
    Load 4ch fused slice if exists; otherwise fallback to stacking single-modality slices.
    Then mask out disallowed modalities by allowed_mask.

    IMPORTANT:
    - This dataset does NOT perform curriculum dropout / gradual missing.
    - present_full is constant per stage (equals allowed_mask); this reflects "what channels are allowed for this stage",
      not "per-sample random missing".
    """
    def __init__(
        self,
        base_ds: Dataset,
        fused_dir: Path,
        allowed_mask: torch.Tensor,
        dtype: str = "float16",
        fallback_dirs: Optional[Dict[str, Path]] = None,
    ):
        self.base = base_ds
        self.fused = fused_dir

        self.allowed = (
            allowed_mask.detach().cpu().numpy()
            if torch.is_tensor(allowed_mask) else np.array(allowed_mask)
        ).astype(np.float32)
        self.dt = np.float16 if dtype == "float16" else np.float32

        self.fallback_dirs = fallback_dirs or {}

    def __len__(self):
        return len(self.base)

    def _load_single_mod(self, mod: str, sid: str) -> Optional[np.ndarray]:
        p = self.fallback_dirs.get(mod, None)
        if p is None:
            return None
        f = p / f"{sid}.npy"
        if not f.exists():
            return None
        a = _np_load(str(f), dtype=np.float32)
        if a.ndim == 2:
            return a
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0]
        if a.ndim == 3 and a.shape[-1] == 1:
            return a[..., 0]
        return a.squeeze()

    def _get_img4(self, sid: str) -> np.ndarray:
        f = self.fused / f"{sid}.npy"
        if f.exists():
            a = _np_load(str(f), dtype=self.dt)
            # accept (4,H,W) or (H,W,4)
            if a.ndim == 3 and a.shape[0] == 4:
                return a
            if a.ndim == 3 and a.shape[-1] == 4:
                return np.transpose(a, (2, 0, 1))
            raise ValueError(f"Bad fused slice shape for {sid}: {a.shape}")

        # fallback: stack single-modality dirs
        xs = []
        H = W = None
        for m in MODS:
            a = self._load_single_mod(m, sid)
            if a is None:
                xs.append(None)
                continue
            if a.ndim != 2:
                a = a.squeeze()
            if H is None:
                H, W = a.shape
            xs.append(a.astype(np.float32, copy=False))

        if H is None:
            raise FileNotFoundError(f"Neither fused nor single-modality slices found for sid={sid}")

        out = np.zeros((4, H, W), dtype=np.float32)
        for i, a in enumerate(xs):
            if a is None:
                continue
            out[i] = a
        return out.astype(self.dt, copy=False)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        sid = str(item["idx"])

        x4 = self._get_img4(sid)  # (4,H,W)
        if x4.ndim != 3 or x4.shape[0] != 4:
            raise ValueError(f"Expect fused (4,H,W), got {x4.shape} for sid={sid}")

        H, W = x4.shape[1], x4.shape[2]
        allow = self.allowed.reshape(4, 1, 1)

        img = (x4.astype(np.float32, copy=False) * allow).reshape(4, H, W)
        present = (self.allowed > 0).astype(np.float32)

        return {
            "image4_full": torch.from_numpy(img),
            "present_full": torch.from_numpy(present),
            "mask": item["mask"],
            "idx": item["idx"],
            "patient": _patient_from_sid(sid),
        }


def _set_seed(seed: int):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: nn.Module, meta: Optional[nn.Module], cfg: StageConfig):
    param_groups = [
        {
            "params": model.parameters(),
            "lr": cfg.base_lr,
            "weight_decay": cfg.weight_decay,
        }
    ]

    if cfg.use_meta and (meta is not None):
        param_groups.append(
            {
                "params": meta.parameters(),
                "lr": cfg.meta_lr,
                "weight_decay": 0.0,  # 建议 meta 不做 wd
            }
        )

    n = cfg.optim_name.lower()
    if n == "adam":
        return optim.Adam(param_groups)
    if n == "adamw":
        return optim.AdamW(param_groups)
    if n == "sgd":
        # 注意：SGD 的 momentum 只对有意义的权重有效；meta 通常不建议 SGD
        return optim.SGD(param_groups, momentum=0.9)
    raise ValueError(cfg.optim_name)



def _build_scheduler(optimizer, cfg: StageConfig):
    warm = int(cfg.max_epoch * 0.1)
    n_groups = len(optimizer.param_groups)

    def _maybe_group_lambdas(lambda_backbone):
        # group0: backbone
        # group1: meta (保持常数)
        if n_groups >= 2 and cfg.use_meta:
            return [lambda_backbone, (lambda e: 1.0)]
        return lambda_backbone

    if cfg.lr_scheduler == "warmupMultistep":
        if cfg.step_num_lr == 2:
            ms = [int(cfg.max_epoch * 0.3), int(cfg.max_epoch * 0.6)]
        elif cfg.step_num_lr == 3:
            ms = [int(cfg.max_epoch * 0.25), int(cfg.max_epoch * 0.4), int(cfg.max_epoch * 0.6)]
        else:
            ms = [
                int(cfg.max_epoch * 0.15),
                int(cfg.max_epoch * 0.35),
                int(cfg.max_epoch * 0.55),
                int(cfg.max_epoch * 0.7),
            ]

        def lr_lambda_backbone(e):
            if e < max(1, warm):
                return (e + 1) / max(1, warm)
            steps = sum(int(m <= e) for m in ms)
            return 0.1 ** steps

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_maybe_group_lambdas(lr_lambda_backbone))

    if cfg.lr_scheduler == "warmupCosine":
        def lr_lambda_backbone(e):
            if e < max(1, warm):
                return (e + 1) / max(1, warm)
            return 0.5 * (math.cos((e - warm) / max(1, cfg.max_epoch - warm) * math.pi) + 1)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_maybe_group_lambdas(lr_lambda_backbone))

    if cfg.lr_scheduler == "autoReduce":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6, verbose=True, cooldown=2, min_lr=0.0
        )

    raise ValueError(cfg.lr_scheduler)



def _gather_best_losses(best_loss_by_name: dict, ddp: bool, world_size: int):
    if not ddp or not dist.is_initialized():
        return best_loss_by_name
    all_dicts = [None for _ in range(world_size)]
    dist.all_gather_object(all_dicts, best_loss_by_name)
    merged = {}
    for d in all_dicts:
        for k, v in d.items():
            if (k not in merged) or (v < merged[k]):
                merged[k] = v
    return merged


def select_by_median(names, losses, keep_ratio: float):
    if len(names) == 0:
        empty = np.array([], dtype=np.float32)
        idx = np.array([], dtype=np.int64)
        return [], empty, idx, empty, empty

    losses = np.asarray(losses, dtype=np.float32)
    N = len(losses)
    K = max(1, int(round(float(keep_ratio) * float(N))))
    med = float(np.median(losses))
    order = np.argsort(np.abs(losses - med), kind="mergesort")
    idx = order[:K]

    sel_names = [names[i] for i in idx]
    sel_losses = losses[idx]
    sel_probs = np.full_like(sel_losses, 1.0 / float(N), dtype=np.float32)
    sel_isw = np.ones_like(sel_losses, dtype=np.float32)
    return sel_names, sel_losses, idx, sel_probs, sel_isw


def run_stage(cfg: StageConfig):
    ddp = cfg.ddp
    rank = int(os.environ.get("RANK", 0)) if ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1

    _set_seed(cfg.seed + rank if ddp else cfg.seed)
    device = _resolve_device(cfg, ddp=ddp, local_rank=local_rank)

    Path(cfg.base_dir).mkdir(parents=True, exist_ok=True)

    auto_first_stage = (cfg.prev_img_modes is None) or (len(cfg.prev_img_modes) == 0)
    is_first_stage = auto_first_stage if (cfg.force_first_stage is None) else cfg.force_first_stage
    is_main = (rank == 0)
    first_batch_logged = False

    fh = None
    root_logger = logging.getLogger()
    if is_main:
        fh = logging.FileHandler(os.path.join(cfg.base_dir, "train.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger.addHandler(fh)

    mode_idx = MODS.index(cfg.img_mode)

    # -------- allowed modalities in fused 4ch input --------
    allowed_vec = torch.zeros(4, dtype=torch.float32)
    prevs_for_4ch = []
    if (not is_first_stage) and cfg.allow_prev_in_4ch:
        disc_prev = _discover_prev_modalities(cfg.img_mode) or []
        prevs_for_4ch = list(set((cfg.prev_img_modes or []) + disc_prev))

    for m in prevs_for_4ch:
        if m in MODS:
            allowed_vec[MODS.index(m)] = 1.0
    allowed_vec[mode_idx] = 1.0

    allowed_idx = [i for i, v in enumerate(allowed_vec.tolist()) if v > 0.5]
    prev_idx_allowed = [i for i in allowed_idx if i != mode_idx]  # for 4ch input
    
    # 对比学习使用所有前序模态的 prototype (不仅是 allowed 的)
    # 这对于防止语义漂移至关重要
    prev_idx_for_contrast = []
    if cfg.prev_img_modes:
        for m in cfg.prev_img_modes:
            if m in MODS:
                prev_idx_for_contrast.append(MODS.index(m))
    prev_idx = prev_idx_for_contrast  # 用于 prototype 对比学习

    if is_main:
        allowed_list = [MODS[i] for i in allowed_idx]
        prev_contrast_list = [MODS[i] for i in prev_idx] if prev_idx else []
        logging.info(f"[Stage] img_mode={cfg.img_mode} | allowed_modalities_for_4ch={allowed_list}")
        logging.info(f"[Stage] prev_modalities_for_contrast={prev_contrast_list}")
        logging.info(f"[Stage] device={device} | ddp={ddp} rank={rank}/{world_size}")

    # -------- datasets & dataloaders --------
    fused_dir = Path(cfg.fused_dir) if cfg.fused_dir else (Path(cfg.data_path) / "BraTS_fusedslice")
    fallback_dirs = {m: Path(cfg.data_path) / f"imgs_{m}" for m in MODS}

    train_ds_base = BaseDataSets(cfg.data_path, "train", cfg.img_mode, "masks_all", cfg.train_list, cfg.images_rate)
    val_ds_base = BaseDataSets(cfg.data_path, "val", cfg.img_mode, "masks_all", cfg.val_list)

    train_ds = FourChFusedDataset(
        train_ds_base, fused_dir, allowed_mask=allowed_vec, dtype=cfg.fused_dtype, fallback_dirs=fallback_dirs
    )
    val_ds = FourChFusedDataset(
        val_ds_base, fused_dir, allowed_mask=allowed_vec, dtype=cfg.fused_dtype, fallback_dirs=fallback_dirs
    )

    def _dl_kwargs(nw: int, pf: int):
        kw = dict(num_workers=nw, pin_memory=True)
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = pf
        return kw

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            drop_last=True,
            **_dl_kwargs(cfg.num_workers_train, cfg.prefetch_train),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **_dl_kwargs(cfg.num_workers_val, cfg.prefetch_val),
        )
    else:
        train_sampler = PatientBatchSampler(train_ds_base.sample_list, cfg.batch_size)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            **_dl_kwargs(cfg.num_workers_train, cfg.prefetch_train),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **_dl_kwargs(cfg.num_workers_val, cfg.prefetch_val),
        )

    # -------- Replay Buffer Loading (经验回放) --------
    replay_ds = None
    replay_loader = None
    mixed_batch_sampler = None
    use_replay_training = False
    
    if cfg.use_replay and (not is_first_stage) and (cfg.prev_img_modes is not None):
        # 加载所有前序模态的 replay buffer
        replay_dict = _load_all_replay_buffers(cfg.prev_img_modes, device="cpu")
        
        if len(replay_dict) > 0:
            # 创建 ReplayDataset
            replay_ds = ReplayDataset(
                replay_dict=replay_dict,
                fused_dir=fused_dir,
                allowed_mask=allowed_vec,
                dtype=cfg.fused_dtype,
            )
            
            # 创建 Replay DataLoader
            replay_samples_per_epoch = int(len(train_ds) * cfg.replay_ratio)
            replay_loader = DataLoader(
                replay_ds,
                batch_size=max(1, int(cfg.batch_size * cfg.replay_ratio)),
                shuffle=True,
                drop_last=True,
                **_dl_kwargs(cfg.num_workers_train // 2, cfg.prefetch_train),
            )
            
            # 创建混合批次采样器
            mixed_batch_sampler = MixedBatchSampler(
                current_dataset_size=len(train_ds),
                replay_dataset=replay_ds,
                batch_size=cfg.batch_size,
                current_ratio=1.0 - cfg.replay_ratio,
                seed=cfg.seed,
            )
            
            use_replay_training = True
            if is_main:
                logging.info(f"[Replay] Enabled replay training with {len(replay_ds)} samples from {list(replay_dict.keys())}")
                logging.info(f"[Replay] replay_ratio={cfg.replay_ratio}, batch_config={mixed_batch_sampler.get_batch_config()}")
        else:
            if is_main:
                logging.info("[Replay] No replay buffers found, skipping replay training")

    # -------- model + projection heads + proto memory --------
    net = CPH(n_classes=3, in_channels=cfg.in_channels).to(device)
    net.train()

    # probe shape to get mid channels
    with torch.no_grad():
        probe = torch.randn(1, cfg.in_channels, 224, 224, device=device)
        out = net(probe, return_feats=True, return_graph=True)
        if isinstance(out, tuple):
            _, aux = out
            C_mid = aux["feat_map"].shape[1]
            if "struct_maps" in aux:
                s2, s3 = aux["struct_maps"]
                Cs = s2.shape[1] + s3.shape[1]
            else:
                Cs = None
        else:
            C_mid, Cs = 3, None

    D = int(getattr(cfg, "proto_dim", 128))
    proj_tumor = nn.Conv2d(C_mid, D, kernel_size=1, bias=False).to(device)
    proj_struct = nn.Conv2d(Cs, D, kernel_size=1, bias=False).to(device) if Cs is not None else None

    # 【修复-严重3】使用单 prototype per class (Kt=1, Ks=1)
    # 论文设计: "one prototype per subregion/class"，用于总结 "globally shared semantics"
    proto_mem = PrototypeMemory(d=D, Kt=1, Ks=1, ema_m=0.05, learnable=False, device=str(device))

    class Bundle(nn.Module):
        def __init__(self, a, pt, ps, c):
            super().__init__()
            self.net = a
            self.proj_tumor = pt
            self.proj_struct = ps if ps is not None else nn.Identity()
            self.proto = c

    bundle = Bundle(net, proj_tumor, proj_struct, proto_mem).to(device)

    # -------- load prototypes from previous stage if available --------
    if cfg.prev_base_dir:
        prev_dir = Path(cfg.prev_base_dir)
        pfile = prev_dir / "prototypes.pt"
        if is_main:
            logging.info(f"[load] trying prototypes from {pfile}")
        if pfile.exists():
            try:
                try:
                    pstate = torch.load(str(pfile), map_location=device, weights_only=True)
                except TypeError:
                    pstate = torch.load(str(pfile), map_location=device)
                bundle.proto.load_state_dict(pstate, strict=False)
                if is_main:
                    logging.info(f"[load] prototypes loaded OK")
            except Exception as e:
                if is_main:
                    logging.warning(f"[load] prototypes load failed: {e}")

    # -------- MetaHyper (真正的 Bi-level Meta-Learning 损失权重控制器) --------
    w_total = cfg.init_tversky_w + cfg.init_imb_w + cfg.init_lambda_proto
    meta = MetaHyper(
        w_tv0=cfg.init_tversky_w,
        w_ft0=cfg.init_imb_w,
        w_proto0=cfg.init_lambda_proto,
        w_total=w_total,
        # 阶段感知参数
        stage_idx=cfg.stage_idx,
        num_stages=cfg.num_stages,
        equal_init=cfg.meta_equal_init,
        # Prototype 下限保护
        proto_floor_start=cfg.proto_floor_start,
        proto_floor_end=cfg.proto_floor_end,
        # Bi-level Meta-Learning 参数
        meta_update_freq=int(getattr(cfg, "meta_update_freq", 10)),
        inner_lr=float(getattr(cfg, "meta_inner_lr", 1e-4)),
        first_order=bool(getattr(cfg, "meta_first_order", True)),
    ).to(device)
    
    # 判断是否使用 unrolled bi-level
    use_unrolled_meta = bool(getattr(cfg, "meta_unrolled", True)) and cfg.use_meta and cfg.meta_bilevel
    
    if is_main:
        stage_info = meta.get_stage_info()
        logging.info(f"[MetaHyper] stage={stage_info['stage_idx']}/{stage_info['num_stages']}, "
                     f"is_first={stage_info['is_first_stage']}, "
                     f"bias={[f'{b:.2f}' for b in stage_info['stage_bias']]}, "
                     f"weights=(tv={stage_info['weights']['w_tv']:.2f}, "
                     f"ft={stage_info['weights']['w_ft']:.2f}, "
                     f"proto={stage_info['weights']['w_proto']:.2f}), "
                     f"entropy={stage_info['weight_entropy']:.3f}")
        if use_unrolled_meta:
            logging.info(f"[MetaHyper] Using UNROLLED bi-level meta-learning: "
                        f"inner_lr={getattr(cfg, 'meta_inner_lr', 1e-4)}, "
                        f"first_order={getattr(cfg, 'meta_first_order', True)}, "
                        f"update_freq={cfg.meta_update_freq}")

    # DDP wrap
    if ddp:
        bundle = DDP(bundle, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        meta = DDP(meta, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = _build_optimizer(bundle, meta, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    
    # ========== 加载前一 stage 的训练状态 (完整状态传递) ==========
    # 这样可以实现跨 stage 的训练连续性，避免从头初始化
    if cfg.prev_base_dir:
        prev_dir = Path(cfg.prev_base_dir)
        
        # 加载 MetaHyper 状态
        meta_state_file = prev_dir / "training_state.pt"
        if meta_state_file.exists() and cfg.use_meta:
            try:
                try:
                    prev_state = torch.load(str(meta_state_file), map_location=device, weights_only=False)
                except TypeError:
                    prev_state = torch.load(str(meta_state_file), map_location=device)
                
                # 加载 MetaHyper 状态 (除了阶段相关的 buffer)
                if "meta" in prev_state and prev_state["meta"] is not None:
                    meta_state = prev_state["meta"]
                    # 过滤掉阶段相关的 buffer，因为新 stage 有新的阶段配置
                    keys_to_skip = ["stage_bias", "proto_floor", "recent_val_loss", "ema_val_loss", "val_loss_count"]
                    filtered_state = {k: v for k, v in meta_state.items() if k not in keys_to_skip}
                    M(meta).load_state_dict(filtered_state, strict=False)
                    if is_main:
                        logging.info(f"[load] MetaHyper state loaded from previous stage (partial, excluding stage-specific buffers)")
                
                # 可选：加载 optimizer 状态 (通常不建议，因为参数空间可能变化)
                # if "optimizer" in prev_state and prev_state["optimizer"] is not None:
                #     optimizer.load_state_dict(prev_state["optimizer"])
                
                if is_main:
                    prev_best = prev_state.get("best_avg3", 0.0)
                    prev_stage = prev_state.get("stage_idx", -1)
                    logging.info(f"[load] Previous stage info: stage_idx={prev_stage}, best_avg3={prev_best:.4f}")
            except Exception as e:
                if is_main:
                    logging.warning(f"[load] training_state load failed: {e}")
        
        # 加载 projections (用于特征提取的一致性)
        proj_file = prev_dir / "projections.pt"
        if proj_file.exists():
            try:
                try:
                    proj_state = torch.load(str(proj_file), map_location=device, weights_only=True)
                except TypeError:
                    proj_state = torch.load(str(proj_file), map_location=device)
                
                if "proj_tumor" in proj_state and proj_state["proj_tumor"] is not None:
                    M(bundle).proj_tumor.load_state_dict(proj_state["proj_tumor"])
                if "proj_struct" in proj_state and proj_state["proj_struct"] is not None:
                    if not isinstance(M(bundle).proj_struct, nn.Identity):
                        M(bundle).proj_struct.load_state_dict(proj_state["proj_struct"])
                if is_main:
                    logging.info(f"[load] projections loaded from previous stage")
            except Exception as e:
                if is_main:
                    logging.warning(f"[load] projections load failed: {e}")

    # -------- patient graph weighter (带 PRM Gate) --------
    weighter = PatientGraphWeighter(
        k=int(getattr(cfg, "patient_graph_k", 5)),
        sigma=float(getattr(cfg, "patient_graph_sigma", 1.0)),
        weighted_H=True,
        a_topo=float(getattr(cfg, "patient_graph_a_topo", 0.5)),
        b_agree=float(getattr(cfg, "patient_graph_b_agree", 0.3)),
        d_conf=float(getattr(cfg, "patient_graph_d_conf", 0.2)),
        kappa=float(getattr(cfg, "patient_graph_kappa", 5.0)),
        mix_uniform=float(getattr(cfg, "patient_graph_mix_uniform", 0.1)),
        # PRM Gate 参数
        gate_ratio=float(getattr(cfg, "prm_gate_ratio", 0.5)),
        gate_min_samples=int(getattr(cfg, "prm_gate_min_samples", 2)),
        gate_score_thresh=float(getattr(cfg, "prm_gate_score_thresh", 0.0)),
    ).to(device)
    weighter.eval()
    for p in weighter.parameters():
        p.requires_grad_(False)
    
    if is_main:
        logging.info(f"[PRM] gate_ratio={cfg.prm_gate_ratio}, gate_min_samples={cfg.prm_gate_min_samples}")

    def M(x):
        return x.module if isinstance(x, DDP) else x

    def _all_gather_cat(x: torch.Tensor):
        if not (ddp and dist.is_initialized()):
            return x
        xs = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(xs, x.contiguous())
        return torch.cat(xs, dim=0)

    @torch.no_grad()
    def _global_has_no_t1ce(present_eff_4: torch.Tensor) -> bool:
        # present_eff_4: [B,4]
        no_ce_local = (present_eff_4[:, 3] < 0.5).to(torch.float32).sum()
        if ddp and dist.is_initialized():
            dist.all_reduce(no_ce_local, op=dist.ReduceOp.SUM)
        return bool(no_ce_local.item() > 0.0)

    # 【修复-严重1】移除 @torch.no_grad() 以恢复 pTAC 梯度流
    # 原问题：pTAC 对比损失的 region features 在 no_grad 下计算，导致梯度断链
    # 修复后：region features 保持梯度，pTAC loss 可以正确反向传播到 backbone
    def _pool_regions_bcd(feat_map: torch.Tensor, masks3_224: torch.Tensor, probs3_224: torch.Tensor):
        """
        Region-wise feature pooling (mask-weighted average)
        
        将 feature map 按照 mask 区域进行加权平均，得到每个区域的表示向量。
        
        Args:
            feat_map:   [B, D, h, w] - 投影后的特征图
            masks3_224: [B, 3, 224, 224] - 三个区域的 mask (WT/TC/ET)
            probs3_224: [B, 3, 224, 224] - 预测概率
            
        Returns:
            f_bcd:    [B, 3, D] - 每个区域的特征向量 (保持梯度)
            avail_bc: [B, 3] - 每个区域是否有效
            conf_bc:  [B, 3] - 每个区域的置信度
            m_ds:     [B, 3, h, w] - 下采样后的 mask
        """
        B, D_, h, w = feat_map.shape
        m_ds = F.adaptive_avg_pool2d(masks3_224.float(), output_size=(h, w))  # [B,3,h,w]
        area_ds = m_ds.flatten(2).sum(-1).clamp_min(1e-6)  # [B,3]
        wgt = m_ds / area_ds.view(B, 3, 1, 1)

        feat_flat = feat_map.flatten(2)  # [B,D,HW]
        wgt_flat = wgt.flatten(2)        # [B,3,HW]
        f_bcd = torch.einsum("bdn,bcn->bcd", feat_flat, wgt_flat)  # [B,3,D] 保持梯度

        area224 = masks3_224.flatten(2).sum(-1)  # [B,3]
        avail_bc = (area224 > 1e-6).to(f_bcd.dtype)
        f_bcd = f_bcd * avail_bc.unsqueeze(-1)

        # conf 计算使用 detach 的 probs，因为 conf 只用于 weighting，不需要梯度
        conf_bc = (probs3_224.detach() * masks3_224).flatten(2).sum(-1) / area224.clamp_min(1e-6)
        conf_bc = conf_bc * avail_bc
        return f_bcd, avail_bc, conf_bc, m_ds

    # proto getter compatibility
    proto_obj = M(bundle).proto
    try:
        _ = proto_obj.get_tumor(0, 0, cond="main")
        _HAS_TUMOR_COND = True
    except TypeError:
        _HAS_TUMOR_COND = False

    try:
        _ = proto_obj.get_struct(0, cond="main")
        _HAS_STRUCT_COND = True
    except TypeError:
        _HAS_STRUCT_COND = False

    def _get_tumor_fast(mi: int, ri: int, cond: str):
        return proto_obj.get_tumor(mi, ri, cond=cond) if _HAS_TUMOR_COND else proto_obj.get_tumor(mi, ri)

    def _get_struct_fast(ri: int, cond: str):
        return proto_obj.get_struct(ri, cond=cond) if _HAS_STRUCT_COND else proto_obj.get_struct(ri)

    def _get_meta_params(ref: torch.Tensor, detach_meta: bool = False):
        """
        获取损失计算所需的参数
        
        - α, β, τ: 静态配置值
        - w_tv, w_ft, w_proto: 从 MetaHyper 动态获取
        
        Returns:
            w_tv, w_ft, w_proto, alpha_tv, beta_tv, tau_val
        """
        # α, β, τ 使用静态配置值
        alpha_tv = ref.new_tensor(cfg.alpha)
        beta_tv = ref.new_tensor(cfg.beta)
        tau_val = ref.new_tensor(cfg.tau)
        
        # 默认权重
        w_tv = ref.new_tensor(cfg.init_tversky_w)
        w_ft = ref.new_tensor(cfg.init_imb_w)
        w_proto = ref.new_tensor(cfg.init_lambda_proto)

        # 从 MetaHyper 获取动态权重
        if cfg.use_meta:
            mod = M(meta)
            try:
                w_tv, w_ft, w_proto = mod.weights()
            except Exception as e:
                if is_main:
                    logging.warning(f"[meta] weights() failed, using defaults: {e}")

        # ---- unify dtype/device ----
        w_tv = w_tv.to(ref.device, ref.dtype)
        w_ft = w_ft.to(ref.device, ref.dtype)
        w_proto = w_proto.to(ref.device, ref.dtype)
        alpha_tv = alpha_tv.to(ref.device, ref.dtype)
        beta_tv = beta_tv.to(ref.device, ref.dtype)
        tau_val = tau_val.to(ref.device, ref.dtype)

        # ---- detach if needed (只对权重生效，α/β/τ 是静态的) ----
        if detach_meta:
            w_tv = w_tv.detach()
            w_ft = w_ft.detach()
            w_proto = w_proto.detach()

        return w_tv, w_ft, w_proto, alpha_tv, beta_tv, tau_val

    best_avg3 = 0.0
    best_loss_by_name = {}

    # -------- training loop --------
    for epoch in range(cfg.max_epoch):
        if ddp and isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        display_epoch = epoch + 1

        # proto_floor schedule
        def _floor_schedule_increasing(e):
            T = max(1, cfg.proto_floor_epochs)
            if e < T:
                t = e / float(T)
                return (1.0 - t) * cfg.proto_floor_start + t * cfg.proto_floor_end
            return cfg.proto_floor_end

        if hasattr(M(meta), "set_proto_floor"):
            M(meta).set_proto_floor(_floor_schedule_increasing(epoch))

        # gain schedule
        def _gain_schedule(e):
            if e < cfg.proto_gain_start:
                return 1.0
            t = min(1.0, (e - cfg.proto_gain_start) / max(1, cfg.proto_gain_span))
            return 1.0 + t * (cfg.proto_gain_max - 1.0)

        warm = 1.0 if cfg.proto_warmup_epochs <= 0 else min(1.0, float(epoch + 1) / float(cfg.proto_warmup_epochs))
        gain = _gain_schedule(epoch)

        M(bundle).train()

        train_loss_sum = 0.0
        train_batches = 0
        train_dice_sum = 0.0
        train_dice_n = 0
        train_WT_sum = 0.0
        train_TC_sum = 0.0
        train_ET_sum = 0.0
        train_class_batches = 0
        
        # Replay training stats
        replay_loss_sum = 0.0
        replay_batches = 0
        
        # Bi-level Meta-Learning stats
        meta_update_count = 0
        meta_loss_sum = 0.0
        
        # Reset meta step counter at each epoch
        if cfg.use_meta and hasattr(M(meta), "reset_step_count"):
            M(meta).reset_step_count()
        
        # 创建 validation iterator 用于 bi-level meta-learning
        val_iter = None
        if cfg.use_meta and getattr(cfg, "meta_bilevel", True):
            val_iter = iter(val_loader)

        # Setup replay iterator for this epoch
        replay_iter = None
        if use_replay_training and replay_loader is not None:
            if mixed_batch_sampler is not None:
                mixed_batch_sampler.set_epoch(epoch)
            replay_iter = iter(replay_loader)

        for batch in train_loader:
            imgs_full = batch["image4_full"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            # present_eff is constant per stage (NO gradual missing / NO teacher-student)
            present_eff = batch["present_full"].to(device=device, dtype=torch.float32, non_blocking=True)  # [B,4]
            if present_eff.ndim == 1:
                present_eff = present_eff.unsqueeze(0).repeat(imgs_full.size(0), 1)
            present_eff = present_eff.clone()
            present_eff[:, mode_idx] = 1.0

            imgs_in = imgs_full * present_eff.view(imgs_full.size(0), 4, 1, 1)

            slice_ids = batch["idx"]

            if is_main and not first_batch_logged:
                counts = present_eff.sum(dim=0).detach().cpu().tolist()
                ch = {m: int(counts[i]) for i, m in enumerate(MODS)}
                logging.info(f"[Stage] first_batch_present_counts: {ch} / bsz={present_eff.size(0)}")
                first_batch_logged = True

            out = M(bundle).net(imgs_in, return_feats=True, return_graph=True)
            if isinstance(out, tuple):
                logits, aux = out
            else:
                logits, aux = out, {}

            probs = torch.sigmoid(logits)

            detach_now = bool(getattr(cfg, "meta_detach", False)) or (
                        epoch < int(getattr(cfg, "meta_freeze_epochs", 0)))
            w_tv, w_ft, w_proto, alpha_tv, beta_tv, tau_val = _get_meta_params(probs, detach_meta=detach_now)

            # supervised losses
            loss_tv = (1.0 - tversky_prob(probs, masks, alpha_tv, beta_tv, smooth=1.0)).mean()
            loss_ft = focal_tversky_prob(probs, masks, alpha_tv, beta_tv, gamma=cfg.gamma, smooth=1.0).mean()

            # tumor features
            if "feat_map" in aux:
                F_mid = aux["feat_map"]
                F_proj = M(bundle).proj_tumor(F_mid)
            else:
                F_proj = M(bundle).proj_tumor(logits)

            # struct features
            use_struct = ("struct_maps" in aux) and (not isinstance(M(bundle).proj_struct, nn.Identity))
            if use_struct:
                s2, s3 = aux["struct_maps"]
                s3u = F.interpolate(s3, size=s2.shape[2:], mode="bilinear", align_corners=False)
                Sin = torch.cat([s2, s3u], dim=1)
                Sproj_full = M(bundle).proj_struct(Sin)
            else:
                Sproj_full = F.avg_pool2d(F_proj, kernel_size=3, stride=1, padding=1)

            # pooled region embeddings once
            f_t_bcd, avail_bc, conf_bc, _ = _pool_regions_bcd(F_proj, masks, probs)
            f_s_bcd, avail_s_bc, _, _ = _pool_regions_bcd(Sproj_full, masks, probs)

            # DDP-consistent cond
            t1ce_allowed = bool(allowed_vec[3].item() > 0.5)
            cond_flag = "no_t1ce" if (t1ce_allowed and _global_has_no_t1ce(present_eff)) else "main"

            # prefetch prototypes
            mi_list = [mode_idx] + prev_idx
            tumorP = {(mi, ri): _get_tumor_fast(mi, ri, cond_flag) for mi in mi_list for ri in (0, 1, 2)}
            structP = {ri: _get_struct_fast(ri, "main") for ri in (0, 1, 2)}

            proto_losses = []

            # tumor branch
            for ridx in (0, 1, 2):
                z = f_t_bcd[:, ridx, :][avail_bc[:, ridx].bool()]
                if z.numel() == 0:
                    continue
                pos = torch.cat([tumorP[(mi, ridx)] for mi in mi_list], dim=0)
                neg_groups = []
                for r2 in (0, 1, 2):
                    if r2 == ridx:
                        continue
                    neg_groups.append(torch.cat([tumorP[(mi, r2)] for mi in mi_list], dim=0))
                proto_losses.append(
                    tv_nce_i2p(z, pos, neg_groups, alpha=alpha_tv, beta=beta_tv, tau=tau_val, smooth=1.0, act="sigmoid")
                )

            # struct branch
            for ridx in (0, 1, 2):
                z = f_s_bcd[:, ridx, :][avail_s_bc[:, ridx].bool()]
                if z.numel() == 0:
                    continue
                pos = structP[ridx]
                neg_groups = [structP[r2] for r2 in (0, 1, 2) if r2 != ridx]
                proto_losses.append(
                    tv_nce_i2p(z, pos, neg_groups, alpha=alpha_tv, beta=beta_tv, tau=tau_val, smooth=1.0, act="sigmoid")
                )

            finite = [l for l in proto_losses if torch.isfinite(l)]
            if len(finite) > 0:
                proto_cl_loss = torch.stack(finite).mean()  # 平均每个 term
                proto_cl_loss = proto_cl_loss / max(1.0, float(imgs_in.size(0)))  # 再按 batch 归一
            else:
                proto_cl_loss = probs.new_tensor(0.0)

            proto_cl_loss = warm * gain * proto_cl_loss

            # -------- Replay Training: 监督损失 + 对比学习 (经验回放完整训练) --------
            # 修复【严重-3】: Replay 数据现在同时参与监督损失和对比学习损失
            replay_cl_loss = probs.new_tensor(0.0)
            replay_sup_loss = probs.new_tensor(0.0)
            
            if use_replay_training and replay_iter is not None:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    # 重置 iterator
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)
                
                # 处理 replay batch
                replay_imgs = replay_batch["image4_full"].to(device=device, dtype=torch.float32, non_blocking=True)
                replay_masks = replay_batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
                replay_present = replay_batch["present_full"].to(device=device, dtype=torch.float32, non_blocking=True)
                replay_mods = replay_batch.get("modality", ["unknown"] * replay_imgs.size(0))
                
                if replay_present.ndim == 1:
                    replay_present = replay_present.unsqueeze(0).repeat(replay_imgs.size(0), 1)
                
                # 前向传播 replay 数据 (不再使用 no_grad，允许梯度传播)
                replay_imgs_in = replay_imgs * replay_present.view(replay_imgs.size(0), 4, 1, 1)
                
                # ========== Replay 前向传播 (有梯度) ==========
                replay_out = M(bundle).net(replay_imgs_in, return_feats=True, return_graph=True)
                if isinstance(replay_out, tuple):
                    replay_logits, replay_aux = replay_out
                else:
                    replay_logits, replay_aux = replay_out, {}
                
                replay_probs = torch.sigmoid(replay_logits)
                
                # ========== Replay 监督损失 (Tversky + Focal Tversky) ==========
                # 这是修复的核心: replay 数据参与分割监督损失，防止对旧模态的分割能力退化
                replay_loss_tv = (1.0 - tversky_prob(replay_probs, replay_masks, alpha_tv, beta_tv, smooth=1.0)).mean()
                replay_loss_ft = focal_tversky_prob(replay_probs, replay_masks, alpha_tv, beta_tv, gamma=cfg.gamma, smooth=1.0).mean()
                
                # replay 监督损失按比例加权
                replay_sup_weight = cfg.replay_ratio  # 使用 replay_ratio 作为监督损失的权重
                replay_sup_loss = replay_sup_weight * (w_tv * replay_loss_tv + w_ft * replay_loss_ft)
                
                # ========== Replay 特征提取 (用于对比学习) ==========
                # 提取 replay 特征
                if "feat_map" in replay_aux:
                    replay_F_mid = replay_aux["feat_map"]
                    replay_F_proj = M(bundle).proj_tumor(replay_F_mid)
                else:
                    replay_F_proj = M(bundle).proj_tumor(replay_logits)
                
                # 结构特征
                if use_struct and "struct_maps" in replay_aux:
                    rs2, rs3 = replay_aux["struct_maps"]
                    rs3u = F.interpolate(rs3, size=rs2.shape[2:], mode="bilinear", align_corners=False)
                    rSin = torch.cat([rs2, rs3u], dim=1)
                    replay_Sproj = M(bundle).proj_struct(rSin)
                else:
                    replay_Sproj = F.avg_pool2d(replay_F_proj, kernel_size=3, stride=1, padding=1)
                
                # 池化 replay 区域特征（保留梯度，使 replay 对比损失可回传到主网络）
                replay_f_t_bcd, replay_avail_bc, replay_conf_bc, _ = _pool_regions_bcd(
                    replay_F_proj, replay_masks, replay_probs
                )
                replay_f_s_bcd, replay_avail_s_bc, _, _ = _pool_regions_bcd(
                    replay_Sproj, replay_masks, replay_probs
                )
                
                # ========== Replay 对比学习：以 prototype 为锚点 ==========
                # 策略：让 replay 样本的特征也与当前模态的原型对比
                # 这样可以保持跨模态的语义一致性，防止灾难性遗忘
                
                replay_proto_losses = []
                
                # Replay tumor branch: 让 replay 样本与所有模态的原型对比
                for ridx in (0, 1, 2):
                    z_replay = replay_f_t_bcd[:, ridx, :][replay_avail_bc[:, ridx].bool()]
                    if z_replay.numel() == 0:
                        continue
                    
                    # 正样本：同区域的所有模态原型 (包括当前和前序)
                    pos = torch.cat([tumorP[(mi, ridx)] for mi in mi_list], dim=0)
                    
                    # 负样本：不同区域的所有模态原型
                    neg_groups = []
                    for r2 in (0, 1, 2):
                        if r2 == ridx:
                            continue
                        neg_groups.append(torch.cat([tumorP[(mi, r2)] for mi in mi_list], dim=0))
                    
                    replay_proto_losses.append(
                        tv_nce_i2p(z_replay, pos, neg_groups, alpha=alpha_tv, beta=beta_tv, 
                                   tau=tau_val, smooth=1.0, act="sigmoid")
                    )
                
                # Replay struct branch
                for ridx in (0, 1, 2):
                    z_replay = replay_f_s_bcd[:, ridx, :][replay_avail_s_bc[:, ridx].bool()]
                    if z_replay.numel() == 0:
                        continue
                    pos = structP[ridx]
                    neg_groups = [structP[r2] for r2 in (0, 1, 2) if r2 != ridx]
                    replay_proto_losses.append(
                        tv_nce_i2p(z_replay, pos, neg_groups, alpha=alpha_tv, beta=beta_tv,
                                   tau=tau_val, smooth=1.0, act="sigmoid")
                    )
                
                # 计算 replay 对比损失
                replay_finite = [l for l in replay_proto_losses if torch.isfinite(l)]
                if len(replay_finite) > 0:
                    replay_cl_loss = torch.stack(replay_finite).mean()
                    replay_cl_loss = replay_cl_loss / max(1.0, float(replay_imgs.size(0)))
                    replay_cl_loss = warm * gain * cfg.replay_contrast_weight * replay_cl_loss
                
                replay_loss_sum += float((replay_cl_loss + replay_sup_loss).detach().item())
                replay_batches += 1

            # total loss: 当前数据损失 + replay 监督损失 + replay 对比损失
            # replay_sup_loss: 防止分割能力退化
            # replay_cl_loss: 防止语义漂移
            loss_total = w_tv * loss_tv + w_ft * loss_ft + w_proto * proto_cl_loss + replay_sup_loss + replay_cl_loss

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            
            # ========== Bi-level Meta-Learning: One-Step Unrolled Update ==========
            # 真正的 bi-level: w -> θ' -> L_val(θ') -> ∇_w
            if cfg.use_meta and getattr(cfg, "meta_bilevel", True) and val_iter is not None:
                if hasattr(M(meta), "should_meta_update") and M(meta).should_meta_update():
                    try:
                        # 获取 validation batch (query batch)
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            val_batch = next(val_iter)
                        
                        val_imgs = val_batch["image4_full"].to(device=device, dtype=torch.float32, non_blocking=True)
                        val_masks = val_batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
                        
                        with torch.enable_grad():
                            # ========== 使用 Unrolled Bi-level Meta-Learning ==========
                            if use_unrolled_meta and hasattr(M(meta), "meta_step_unrolled"):
                                # Support batch = 当前 training batch
                                # Query batch = validation batch
                                meta_loss, meta_stats = M(meta).meta_step_unrolled(
                                    model=M(bundle).net,
                                    support_imgs=imgs_in.detach(),  # detach 避免影响主训练
                                    support_masks=masks,
                                    query_imgs=val_imgs,
                                    query_masks=val_masks,
                                    alpha=cfg.alpha,
                                    beta=cfg.beta,
                                    gamma=cfg.gamma,
                                    proto_loss_sup=proto_cl_loss.detach() if torch.isfinite(proto_cl_loss) else None,
                                )
                            else:
                                # Fallback: 旧的 meta_step (不推荐)
                                val_logits = M(bundle).net(val_imgs)
                                val_probs = torch.sigmoid(val_logits)
                                meta_loss, meta_stats = M(meta).meta_step(
                                    query_probs=val_probs,
                                    query_masks=val_masks,
                                    alpha=cfg.alpha,
                                    beta=cfg.beta,
                                    gamma=cfg.gamma,
                                    proto_loss=proto_cl_loss.detach() if torch.isfinite(proto_cl_loss) else None,
                                )
                            
                            # ========== 只更新 MetaHyper 的参数 ==========
                            optimizer.zero_grad(set_to_none=True)
                            meta_loss.backward()
                            
                            # ========== Sanity Check: logits 梯度非零 ==========
                            logits_grad_norm = 0.0
                            if hasattr(M(meta), "logits") and M(meta).logits.grad is not None:
                                logits_grad_norm = float(M(meta).logits.grad.norm().item())
                            
                            # 只对 meta 参数组进行更新 (清除主网络梯度)
                            if len(optimizer.param_groups) >= 2:
                                for param in optimizer.param_groups[0]["params"]:
                                    param.grad = None
                            
                            optimizer.step()
                            
                            meta_update_count += 1
                            meta_loss_sum += float(meta_loss.detach().item())
                            
                            # ========== 日志: 每次 meta update 输出详细信息 ==========
                            if is_main and (meta_update_count == 1 or meta_update_count % 10 == 0):
                                w_tv_now = meta_stats.get("w_tv", 0)
                                w_ft_now = meta_stats.get("w_ft", 0)
                                w_proto_now = meta_stats.get("w_proto", 0)
                                entropy = meta_stats.get("weight_entropy", 0)
                                L_val = meta_stats.get("L_val_dice", meta_stats.get("val_total", 0))
                                logging.info(
                                    f"[Meta#{meta_update_count}] "
                                    f"w=[{w_tv_now:.3f}, {w_ft_now:.3f}, {w_proto_now:.3f}], "
                                    f"entropy={entropy:.3f}, "
                                    f"L_val={L_val:.4f}, "
                                    f"grad_norm={logits_grad_norm:.4f}"
                                )
                                
                                # 权重塌缩警告
                                if "warning" in meta_stats:
                                    logging.warning(f"[Meta] {meta_stats['warning']}")
                                
                                # 梯度为零警告
                                if logits_grad_norm < 1e-8:
                                    logging.warning(f"[Meta] logits gradient is near zero! Check computation graph.")
                                    
                    except Exception as e:
                        if is_main and train_batches == 1:
                            logging.warning(f"[Meta] bi-level update failed: {e}")
                            import traceback
                            traceback.print_exc()

            # ---- hypergraph-weighted prototype EMA update (with gate) ----
            # PRM 核心: patient-level gate + reliability-weighted EMA
            with torch.no_grad():
                # ========== 修复 H2: z_patient 来自 encoder (hypergraph-fused features) ==========
                # 论文要求: patient embedding 来自 hypergraph-fused encoder features (E4/E5)
                # 而非 decoder 特征
                if use_struct and "struct_maps" in aux:
                    # 使用 struct_maps (来自 HyperEncoder 的 E4/E5 层)
                    s2_enc, s3_enc = aux["struct_maps"]
                    # 全局平均池化得到 patient-level embedding
                    z_enc_2 = s2_enc.flatten(2).mean(-1)  # [B, C_s2]
                    z_enc_3 = s3_enc.flatten(2).mean(-1)  # [B, C_s3]
                    z_local = torch.cat([z_enc_2, z_enc_3], dim=1)  # [B, C_s2+C_s3]
                else:
                    # Fallback: 使用投影后的 decoder 特征
                    z_local = F_proj.flatten(2).mean(-1)  # [B, D]
                
                z_all = _all_gather_cat(z_local)

                # detach region features 用于 prototype 更新（不需要梯度）
                f_t_all = _all_gather_cat(f_t_bcd.detach())
                f_s_all = _all_gather_cat(f_s_bcd.detach())
                avail_all = _all_gather_cat(avail_bc)
                conf_all = _all_gather_cat(conf_bc)

                # ========== PRM Core: patient-level gate + reliability weighting ==========
                # omega: [N, C] 权重矩阵 (每列和为 1)
                # gate_mask: [N, C] bool 矩阵 (patient-level gate 展开到 class)
                omega_t, gate_mask_t = weighter(z_patient=z_all, f_cls=f_t_all, conf_cls=conf_all, return_gate_mask=True)
                omega_s, gate_mask_s = weighter(z_patient=z_all, f_cls=f_s_all, conf_cls=conf_all, return_gate_mask=True)

                def weighted_mean_per_class(omega: torch.Tensor, f_all: torch.Tensor, 
                                            avail_all_: torch.Tensor, gate_mask: torch.Tensor):
                    """
                    计算 gate-weighted mean for prototype update
                    
                    公式: \tilde{P}_c = Σ_n ω_{n,c} * g_n * \hat{P}_{n,c}
                    其中 g_n 是 patient-level gate, ω_{n,c} 是 reliability weight
                    
                    只有通过 hypergraph gate 且区域有效的 representative samples 参与 prototype 更新，
                    抑制 outliers，减少 semantic overwriting。
                    
                    Args:
                        omega: [N, C] 权重 (已经由 PatientGraphWeighter 内部应用了 gate)
                        f_all: [N, C, D] 特征 (proposals)
                        avail_all_: [N, C] 区域可用性 (GT 面积 > 0)
                        gate_mask: [N, C] gate 通过标记 (patient-level)
                    
                    Returns:
                        g: [C, D] 加权平均后的 prototype 更新值 (L2 normalized)
                        valid: [C] 每个区域是否有有效更新
                    """
                    # 显式应用 gate_mask 和 avail_mask
                    # 只有 gated=True 且 avail=True 的 (n,c) 才参与
                    omega_gated = omega * gate_mask.float()
                    omega_eff = omega_gated * avail_all_
                    
                    denom = omega_eff.sum(dim=0)  # [C]
                    valid = denom > 1e-6
                    omega_eff = omega_eff / denom.clamp_min(1e-6).unsqueeze(0)  # 重新归一化到 gated+valid set
                    
                    # 加权平均
                    g = torch.einsum("nc,ncd->cd", omega_eff, f_all)  # [C, D]
                    
                    # ========== 修复 H3: L2 normalize 再 EMA ==========
                    # 论文要求: 对聚合结果 L2 normalize 后再进行 EMA 更新
                    g = F.normalize(g, dim=-1, eps=1e-6)
                    
                    return g, valid

                g_t, valid_t = weighted_mean_per_class(omega_t, f_t_all, avail_all, gate_mask_t)
                g_s, valid_s = weighted_mean_per_class(omega_s, f_s_all, avail_all, gate_mask_s)
                
                # 记录 gate 统计 (每个 epoch 第一个 batch)
                if is_main and train_batches == 1:
                    gate_stats_t = weighter.get_gate_stats(gate_mask_t)
                    n_gated = gate_mask_t[:, 0].sum().item()  # patient-level 所有 class 相同
                    logging.info(f"[PRM Gate] {int(n_gated)} / {gate_stats_t['total_samples']} patients passed (patient-level gate)")

                for ridx in (0, 1, 2):
                    if bool(valid_t[ridx].item()):
                        if _HAS_TUMOR_COND:
                            M(bundle).proto.ema_update_tumor(mode_idx, ridx, g_t[ridx:ridx + 1], assign="hard", cond=cond_flag)
                        else:
                            M(bundle).proto.ema_update_tumor(mode_idx, ridx, g_t[ridx:ridx + 1], assign="hard")

                    if bool(valid_s[ridx].item()):
                        if _HAS_STRUCT_COND:
                            M(bundle).proto.ema_update_struct(ridx, g_s[ridx:ridx + 1], assign="hard", cond=cond_flag)
                        else:
                            M(bundle).proto.ema_update_struct(ridx, g_s[ridx:ridx + 1], assign="hard")

            # metrics
            pred = (probs > 0.5).float()
            dsum, ncnt = batch_dice(pred.detach().cpu(), masks.detach().cpu())
            train_dice_sum += float(dsum)
            train_dice_n += int(ncnt)

            p_np = pred.detach().cpu().numpy().astype("uint8")
            t_np = masks.detach().cpu().numpy().astype("uint8")
            d0, _, _ = dice_all(p_np[:, 0], t_np[:, 0])
            d1, _, _ = dice_all(p_np[:, 1], t_np[:, 1])
            d2, _, _ = dice_all(p_np[:, 2], t_np[:, 2])
            train_WT_sum += float(d0)
            train_TC_sum += float(d1)
            train_ET_sum += float(d2)
            train_class_batches += 1

            train_loss_sum += float(loss_total.detach().item())
            train_batches += 1

            # track best supervised loss per slice (for replay)
            with torch.no_grad():
                B = imgs_in.shape[0]
                for b in range(B):
                    lt = 1.0 - tversky_prob(probs[b:b + 1], masks[b:b + 1], alpha_tv, beta_tv, smooth=1.0)
                    lf = focal_tversky_prob(probs[b:b + 1], masks[b:b + 1], alpha_tv, beta_tv, gamma=cfg.gamma, smooth=1.0)
                    sup_loss = float((w_tv * lt + w_ft * lf).mean().item())
                    name = str(slice_ids[b])
                    prev = best_loss_by_name.get(name)
                    if (prev is None) or (sup_loss < prev):
                        best_loss_by_name[name] = sup_loss

        # reduce train stats
        if ddp and dist.is_initialized():
            buf = torch.tensor(
                [train_loss_sum, train_batches, train_dice_sum, train_dice_n,
                 train_WT_sum, train_TC_sum, train_ET_sum, train_class_batches],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            (train_loss_sum, train_batches, train_dice_sum, train_dice_n,
             train_WT_sum, train_TC_sum, train_ET_sum, train_class_batches) = buf.tolist()

        train_loss = train_loss_sum / max(1.0, train_batches)
        train_dice_micro = (train_dice_sum / max(1.0, train_dice_n)) if train_dice_n > 0 else 0.0
        train_WT_avg = train_WT_sum / max(1.0, train_class_batches)
        train_TC_avg = train_TC_sum / max(1.0, train_class_batches)
        train_ET_avg = train_ET_sum / max(1.0, train_class_batches)
        train_dice_avg = (train_WT_avg + train_TC_avg + train_ET_avg) / 3.0

        # -------- validation --------
        M(bundle).eval()
        val_loss_sum = 0.0
        val_batches = 0.0
        val_dice_sum = 0.0
        val_dice_n = 0.0
        WT_sum = 0.0
        TC_sum = 0.0
        ET_sum = 0.0
        class_batches = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs_full = batch["image4_full"].to(device=device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

                logits = M(bundle).net(imgs_full)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()

                dsum, ncnt = batch_dice(pred.detach().cpu(), masks.detach().cpu())
                val_dice_sum += float(dsum)
                val_dice_n += float(ncnt)

                alpha_v = probs.new_tensor(cfg.alpha)
                beta_v = probs.new_tensor(cfg.beta)
                loss_val_tv = (1.0 - tversky_prob(probs, masks, alpha_v, beta_v, smooth=1.0)).mean()
                loss_val_ft = focal_tversky_prob(probs, masks, alpha_v, beta_v, gamma=cfg.gamma, smooth=1.0).mean()
                val_loss_sum += float((cfg.init_tversky_w * loss_val_tv + cfg.init_imb_w * loss_val_ft).item())
                val_batches += 1.0

                p_np = pred.detach().cpu().numpy().astype("uint8")
                t_np = masks.detach().cpu().numpy().astype("uint8")
                d0, _, _ = dice_all(p_np[:, 0], t_np[:, 0])
                d1, _, _ = dice_all(p_np[:, 1], t_np[:, 1])
                d2, _, _ = dice_all(p_np[:, 2], t_np[:, 2])
                WT_sum += float(d0)
                TC_sum += float(d1)
                ET_sum += float(d2)
                class_batches += 1.0

        if ddp and dist.is_initialized():
            buf = torch.tensor(
                [val_loss_sum, val_batches, val_dice_sum, val_dice_n, WT_sum, TC_sum, ET_sum, class_batches],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            (val_loss_sum, val_batches, val_dice_sum, val_dice_n, WT_sum, TC_sum, ET_sum, class_batches) = buf.tolist()

        val_loss = val_loss_sum / max(1.0, val_batches)
        val_dice_micro = (val_dice_sum / max(1.0, val_dice_n)) if val_dice_n > 0 else 0.0
        WT_avg = WT_sum / max(1.0, class_batches)
        TC_avg = TC_sum / max(1.0, class_batches)
        ET_avg = ET_sum / max(1.0, class_batches)
        val_dice_avg = (WT_avg + TC_avg + ET_avg) / 3.0

        # scheduler step
        if cfg.lr_scheduler == "autoReduce":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # keep meta lr constant under ReduceLROnPlateau
        if cfg.lr_scheduler == "autoReduce" and cfg.use_meta and len(optimizer.param_groups) >= 2:
            optimizer.param_groups[1]["lr"] = cfg.meta_lr

        # compute replay loss average
        replay_loss_avg = replay_loss_sum / max(1.0, replay_batches) if replay_batches > 0 else 0.0

        if is_main:
            lr_main = float(optimizer.param_groups[0]["lr"])
            lr_meta = lr_main
            if cfg.use_meta and len(optimizer.param_groups) >= 2:
                lr_meta = float(optimizer.param_groups[1]["lr"])

            # Base log message
            log_msg = (
                f"[{cfg.img_mode}] Epoch {display_epoch:03d}/{cfg.max_epoch:03d} | "
                f"train_loss={train_loss:.4f} train_dice_micro={train_dice_micro:.4f} "
                f"train_dice_avg={train_dice_avg:.4f} (WT={train_WT_avg:.4f} TC={train_TC_avg:.4f} ET={train_ET_avg:.4f}) | "
                f"val_loss={val_loss:.4f} val_dice_micro={val_dice_micro:.4f} "
                f"val_dice_avg={val_dice_avg:.4f} (WT={WT_avg:.4f} TC={TC_avg:.4f} ET={ET_avg:.4f}) | "
                f"lr={lr_main:.6f} meta_lr={lr_meta:.6f}"
            )
            
            # Add replay info if enabled
            if use_replay_training and replay_batches > 0:
                log_msg += f" | replay_cl_loss={replay_loss_avg:.4f} replay_batches={int(replay_batches)}"
            
            logging.info(log_msg)

        # save best model
        if is_main and val_dice_avg > best_avg3:
            best_avg3 = val_dice_avg
            torch.save(M(bundle).net.state_dict(), os.path.join(cfg.base_dir, "model_CPH_best.pth"))
            logging.info(f"[Best] epoch={display_epoch} val_dice_avg={best_avg3:.4f} val_dice_micro={val_dice_micro:.4f}")

    # -------- after training: build replay & save artifacts --------
    if ddp and dist.is_initialized():
        dist.barrier()

    merged = _gather_best_losses(best_loss_by_name, ddp=ddp and dist.is_initialized(), world_size=world_size)

    if is_main:
        # build replay buffer
        try:
            if len(merged) > 0:
                names = list(merged.keys())
                losses = np.array([merged[n] for n in names], dtype=np.float32)
                keep_ratio = float(cfg.p_keep)

                if cfg.per_enable:
                    per_alpha = float(getattr(cfg, "per_alpha", 0.6))
                    per_beta = float(getattr(cfg, "per_beta", 0.4))
                    sel_names, sel_losses, sel_idx, sel_probs, sel_isw = select_with_per(
                        names, losses, keep_ratio=keep_ratio, alpha=per_alpha, beta=per_beta,
                        epsilon=1e-6, seed=42
                    )
                    replay_tag = f"PER(alpha={per_alpha:.2f}, beta={per_beta:.2f})"
                else:
                    sel_names, sel_losses, sel_idx, sel_probs, sel_isw = select_by_median(names, losses, keep_ratio=keep_ratio)
                    replay_tag = "MedianLoss"

                img_dir = Path(cfg.data_path) / f"imgs_{cfg.img_mode}"
                mask_dir = Path(cfg.data_path) / "masks_all"

                def _to_chw(a: np.ndarray):
                    if a.ndim == 2:
                        return a[None, ...]
                    return a

                X_chunks, Y_chunks = [], []
                CHUNK = 1024
                for s in range(0, len(sel_names), CHUNK):
                    part = sel_names[s:s + CHUNK]
                    imgs_np = [_to_chw(np.load(img_dir / f"{n}.npy", allow_pickle=False)) for n in part]
                    masks_np = [_to_chw(np.load(mask_dir / f"{n}.npy", allow_pickle=False)) for n in part]
                    X_chunks.append(torch.from_numpy(np.stack(imgs_np)).to(torch.float16))
                    Y_chunks.append(torch.from_numpy(np.stack(masks_np)).to(torch.uint8))

                Xs = torch.cat(X_chunks, dim=0).contiguous()
                Ys = torch.cat(Y_chunks, dim=0).contiguous()
                Ls = torch.from_numpy(sel_losses.astype(np.float32)).to(torch.float32)
                pats = [_patient_from_sid(n) for n in sel_names]
                mods = [cfg.img_mode] * len(sel_names)

                _save_replay(cfg.img_mode, Xs, Ys, Ls, pats, mods)
                logging.info(f"[Replay][{replay_tag}] unique={len(names)} keep%={cfg.p_keep:.2f} -> saved={len(sel_names)}")
        except Exception as e:
            logging.warning(f"[Replay] failed to build/save replay: {e}")

        # save prototypes
        try:
            torch.save(M(bundle).proto.state_dict(), os.path.join(cfg.base_dir, "prototypes.pt"))
            logging.info(f"[save] prototypes -> {os.path.join(cfg.base_dir, 'prototypes.pt')}")
        except Exception as e:
            logging.warning(f"[save] prototypes failed: {e}")

        # save projections
        try:
            torch.save(
                {
                    "proj_tumor": M(bundle).proj_tumor.state_dict(),
                    "proj_struct": (M(bundle).proj_struct.state_dict() if not isinstance(M(bundle).proj_struct, nn.Identity) else None),
                },
                os.path.join(cfg.base_dir, "projections.pt"),
            )
            logging.info(f"[save] projections -> {os.path.join(cfg.base_dir, 'projections.pt')}")
        except Exception as e:
            logging.warning(f"[save] projections failed: {e}")
        
        # ========== 完整训练状态保存 (用于跨 stage 传递) ==========
        # 保存 optimizer, scheduler, meta 状态，确保训练连续性
        try:
            full_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                "meta": M(meta).state_dict() if cfg.use_meta else None,
                "epoch": cfg.max_epoch,
                "best_avg3": best_avg3,
                "stage_idx": cfg.stage_idx,
                "img_mode": cfg.img_mode,
            }
            torch.save(full_state, os.path.join(cfg.base_dir, "training_state.pt"))
            logging.info(f"[save] training_state -> {os.path.join(cfg.base_dir, 'training_state.pt')}")
        except Exception as e:
            logging.warning(f"[save] training_state failed: {e}")
        except Exception as e:
            logging.warning(f"[save] projections failed: {e}")

        # save last model
        try:
            torch.save(M(bundle).net.state_dict(), os.path.join(cfg.base_dir, "model_CPH_last.pth"))
        except Exception as e:
            logging.warning(f"[save] model_CPH_last failed: {e}")

        # remove file handler
        if fh is not None:
            try:
                root_logger.removeHandler(fh)
                fh.close()
            except Exception:
                pass
