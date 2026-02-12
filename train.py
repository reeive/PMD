# -*- coding: utf-8 -*-
"""
PMD: Prototype-guided Meta DIL for Missing-Modality Brain Tumor Segmentation
Training script - launches stage-by-stage incremental training.

Paper: "From Semantic Drift to Objective Miscalibration: Domain Incremental
        Brain Tumor Segmentation with Missing Modality" (MedIA 2026)

Default hyperparameters match Section 4.2 of the paper:
  - Adam, lr=1e-4, weight_decay=4e-4, batch_size=128, 200 epochs/stage
  - Tversky (α, β, γ) = (0.7, 1.5, 1.2), pTAC τ=0.1
  - PRM: top-q=0.5, power iteration T=10, EMA η=0.99
  - Meta Controller: η_inner=1e-4, η_meta=1e-3, update every K=10 steps
"""
import os
import re
import sys
import argparse
import logging

from utils.stage_driver import StageConfig, run_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def parse_args():
    p = argparse.ArgumentParser(description="PMD: Prototype-guided Meta DIL trainer")

    # ---- Paths & stage lists ----
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--stages", type=str, default="t1,t2,flair,t1ce",
                   help="Comma-separated modality order. "
                        "Clinical: t1,t2,flair,t1ce; Feature: flair,t1ce,t1,t2")
    p.add_argument("--train_lists", type=str, default="")
    p.add_argument("--val_lists", type=str, default="")
    p.add_argument("--train_fmt", type=str, default="lists/train.list")
    p.add_argument("--val_fmt", type=str, default="lists/val.list")

    # ---- Training basics (paper Sec 4.2) ----
    p.add_argument("--max_epoch", type=int, default=200,
                   help="Epochs per stage (paper: 200)")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size (paper: 128)")
    p.add_argument("--images_rate", type=float, default=1.0)

    # ---- Optimizer & LR (paper Sec 4.2: Adam, lr=1e-4, wd=4e-4) ----
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=4e-4)
    p.add_argument("--optim_name", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr_scheduler", type=str, default="warmupMultistep",
                   choices=["warmupMultistep", "warmupCosine", "autoReduce"])
    p.add_argument("--step_num_lr", type=int, default=4)

    # ---- Tversky / Focal parameters (paper Sec 4.2: α=0.7, β=1.5, γ=1.2) ----
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Tversky α (FP weight, paper: 0.7)")
    p.add_argument("--beta", type=float, default=1.5,
                   help="Tversky β (FN weight, paper: 1.5)")
    p.add_argument("--gamma", type=float, default=1.2,
                   help="Focal Tversky γ (paper: 1.2)")

    # ---- pTAC temperature (paper Sec 4.2: τ=0.1) ----
    p.add_argument("--tau", type=float, default=0.1,
                   help="pTAC contrastive temperature (paper: 0.1)")

    # ---- System & seed ----
    p.add_argument("--in_channels", type=int, default=4)
    p.add_argument("--workers_train", type=int, default=8)
    p.add_argument("--workers_val", type=int, default=4)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--ddp", action="store_true")

    # ---- Meta Controller (paper Sec 3.4 & 4.2: η_inner=1e-4, η_meta=1e-3) ----
    p.add_argument("--use_meta", action="store_true",
                   help="Enable Meta Controller for adaptive loss reweighting")
    p.add_argument("--meta_lr", type=float, default=1e-3,
                   help="Meta Controller learning rate (paper: η_meta=1e-3)")
    p.add_argument("--meta_update_freq", type=int, default=10,
                   help="Meta update period K (every K steps)")
    p.add_argument("--meta_inner_lr", type=float, default=1e-4,
                   help="Virtual inner step lr (paper: η_inner=1e-4)")

    # ---- Loss weight init (for MetaHyper w_total computation) ----
    p.add_argument("--init_tversky_w", type=float, default=7.0)
    p.add_argument("--init_imb_w", type=float, default=8.0)
    p.add_argument("--init_lambda_proto", type=float, default=3.5)

    # ---- Prototype scheduling ----
    p.add_argument("--proto_warmup_epochs", type=int, default=10)
    p.add_argument("--proto_floor_start", type=float, default=0.10)
    p.add_argument("--proto_floor_end", type=float, default=0.40)
    p.add_argument("--proto_floor_epochs", type=int, default=60)
    p.add_argument("--proto_align_lambda", type=float, default=5e-3)
    p.add_argument("--proto_target_bias", type=float, default=0.0)
    p.add_argument("--proto_gain_start", type=int, default=30)
    p.add_argument("--proto_gain_span", type=int, default=40)
    p.add_argument("--proto_gain_max", type=float, default=2.0)

    # ---- EMA teacher ----
    p.add_argument("--new_teacher_is_ema", type=int, default=1, choices=[0, 1])
    p.add_argument("--new_teacher_ema_m", type=float, default=0.99)

    # ---- Replay buffer (paper Sec 3.2: retain 10% per stage) ----
    p.add_argument("--mem_size", type=int, default=32000)
    p.add_argument("--p_keep", type=float, default=0.10,
                   help="Fraction of samples kept per stage for replay (paper: 10%%)")

    # ---- PER (Prioritized Experience Replay) ----
    p.add_argument("--per_enable", type=int, default=1, choices=[0, 1])
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta", type=float, default=0.4)
    p.add_argument("--per_epsilon", type=float, default=1e-6)
    p.add_argument("--per_use_ema", type=int, default=1, choices=[0, 1])
    p.add_argument("--per_ema_m", type=float, default=0.7)

    # ---- Stage control ----
    p.add_argument("--force_first_stage", type=int, default=-1, choices=[-1, 0, 1])
    p.add_argument("--allow_prev_in_4ch", type=int, default=1, choices=[0, 1])

    # ---- Replay training (paper Sec 3.2: 1:1 current/replay ratio) ----
    p.add_argument("--use_replay", type=int, default=1, choices=[0, 1])
    p.add_argument("--replay_ratio", type=float, default=0.5,
                   help="Replay batch ratio (paper: 1:1 → 0.5)")
    p.add_argument("--replay_contrast_weight", type=float, default=1.0)
    p.add_argument("--replay_stratified", type=int, default=1, choices=[0, 1])

    # ---- Legacy (kept for backward compatibility) ----
    p.add_argument("--tversky_w", type=float, default=7.0)
    p.add_argument("--imb_w", type=float, default=8.0)
    p.add_argument("--nce_weight", type=float, default=3.5)
    p.add_argument("--ab_freeze_epochs", type=int, default=10)
    p.add_argument("--ab_kl", type=float, default=1e-3)

    return p.parse_args()


def _expand_lists(stages, lists_arg: str, fmt: str):
    if lists_arg.strip():
        parts = [s.strip() for s in lists_arg.split(",") if s.strip()]
        if len(parts) == 1:
            return parts * len(stages)
        assert len(parts) == len(stages)
        return parts
    else:
        if "{mod}" in fmt:
            return [fmt.format(mod=m) for m in stages]
        else:
            return [fmt] * len(stages)


def _resolve_lists(stages, train_lists_arg, val_lists_arg, train_fmt, val_fmt):
    train_lists = _expand_lists(stages, train_lists_arg, train_fmt)
    val_lists = _expand_lists(stages, val_lists_arg, val_fmt)
    return train_lists, val_lists


def _verify_paths(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            logging.error(f"[List Missing] {p}")
        raise FileNotFoundError("Some list files do not exist. See errors above.")


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)

    # ---- DDP init ----
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        import torch
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = (rank == 0)
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main = True

    logging.info(f"[Device] {device}")

    # ---- Resolve stages & lists ----
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    assert len(stages) >= 1

    train_lists, val_lists = _resolve_lists(stages, args.train_lists, args.val_lists,
                                            args.train_fmt, args.val_fmt)
    if is_main:
        _verify_paths(train_lists + val_lists)

    os.makedirs(args.out_root, exist_ok=True)

    seen = []
    prev_base_dir = None

    for i, mod in enumerate(stages):
        base_dir = os.path.join(args.out_root, f"res-{mod}")
        prev_img_modes = seen.copy() if seen else None

        cfg = StageConfig(
            base_dir=base_dir,
            data_path=args.data_path,
            train_list=train_lists[i],
            val_list=val_lists[i],
            img_mode=mod,
            prev_img_modes=prev_img_modes,
            prev_base_dir=prev_base_dir,

            # Replay buffer
            mem_size=args.mem_size,
            p_keep=args.p_keep,

            # Training schedule (paper: 200 epochs, batch 128)
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            images_rate=args.images_rate,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            optim_name=args.optim_name,
            lr_scheduler=args.lr_scheduler,
            step_num_lr=args.step_num_lr,

            # Loss weights (static fallback)
            tversky_w=args.tversky_w,
            imb_w=args.imb_w,
            nce_weight=args.nce_weight,

            # Tversky parameters (paper: α=0.7, β=1.5, γ=1.2)
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,

            # pTAC temperature (paper: τ=0.1)
            tau=args.tau,

            # System
            in_channels=args.in_channels,
            num_workers_train=args.workers_train,
            num_workers_val=args.workers_val,
            device=device,
            seed=args.seed,

            # Meta Controller (paper: η_inner=1e-4, η_meta=1e-3, K=10)
            use_meta=args.use_meta,
            meta_lr=args.meta_lr,
            init_lambda_proto=args.init_lambda_proto,
            init_tversky_w=args.init_tversky_w,
            init_imb_w=args.init_imb_w,
            proto_warmup_epochs=args.proto_warmup_epochs,
            ddp=args.ddp,

            ab_freeze_epochs=args.ab_freeze_epochs,
            ab_kl=args.ab_kl,

            # Prototype scheduling
            proto_floor_start=args.proto_floor_start,
            proto_floor_end=args.proto_floor_end,
            proto_floor_epochs=args.proto_floor_epochs,
            proto_align_lambda=args.proto_align_lambda,
            proto_target_bias=args.proto_target_bias,
            proto_gain_start=args.proto_gain_start,
            proto_gain_span=args.proto_gain_span,
            proto_gain_max=args.proto_gain_max,

            # EMA teacher
            new_teacher_is_ema=bool(args.new_teacher_is_ema),
            new_teacher_ema_m=args.new_teacher_ema_m,

            # PER
            per_enable=bool(args.per_enable),
            per_alpha=args.per_alpha,
            per_beta=args.per_beta,
            per_epsilon=args.per_epsilon,
            per_use_ema=bool(args.per_use_ema),
            per_ema_m=args.per_ema_m,

            force_first_stage=(None if args.force_first_stage < 0 else bool(args.force_first_stage)),
            allow_prev_in_4ch=bool(args.allow_prev_in_4ch),

            # Replay training (paper: 1:1 current/replay)
            use_replay=bool(args.use_replay),
            replay_ratio=args.replay_ratio,
            replay_contrast_weight=args.replay_contrast_weight,
            replay_stratified=bool(args.replay_stratified),

            # Stage-aware Meta
            stage_idx=i,
            num_stages=len(stages),
            meta_equal_init=True,
        )

        if is_main:
            logging.info(f"[Stage {i+1}/{len(stages)}] modality={mod}")
            if prev_img_modes:
                logging.info(f"  prev_modalities: {prev_img_modes}")
            logging.info(f"  alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, tau={args.tau}")
            logging.info(f"  lr={args.base_lr}, wd={args.weight_decay}, batch={args.batch_size}, epochs={args.max_epoch}")
            if args.use_meta:
                logging.info(f"  [Meta Controller] lr_meta={args.meta_lr}, lr_inner={args.meta_inner_lr}, K={args.meta_update_freq}")

        run_stage(cfg)

        seen.append(mod)
        prev_base_dir = base_dir

    if args.ddp:
        import torch.distributed as dist
        dist.barrier()
        if is_main:
            logging.info("\n[All Done] All stages finished successfully.")
        dist.destroy_process_group()
    else:
        logging.info("\n[All Done] All stages finished successfully.")


if __name__ == "__main__":
    main()
