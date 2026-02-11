# -*- coding: utf-8 -*-
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
    p = argparse.ArgumentParser(description="ReHyDIL++ (Proto + Meta + TAC) trainer")

    # ---- Paths & stage lists ----
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--stages", type=str, default="t1,t2,flair,t1ce")
    p.add_argument("--train_lists", type=str, default="")
    p.add_argument("--val_lists", type=str, default="")
    p.add_argument("--train_fmt", type=str, default="lists/train.list")
    p.add_argument("--val_fmt", type=str, default="lists/val.list")

    # ---- Training basics ----
    p.add_argument("--max_epoch", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--images_rate", type=float, default=1.0)

    # ---- Optim & LR ----
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--optim_name", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr_scheduler", type=str, default="warmupMultistep",
                   choices=["warmupMultistep", "warmupCosine", "autoReduce"])
    p.add_argument("--step_num_lr", type=int, default=4)

    # ---- Loss weights (static init for MetaHyper) ----
    p.add_argument("--tversky_w", type=float, default=7.0)
    p.add_argument("--imb_w", type=float, default=8.0)
    p.add_argument("--nce_weight", type=float, default=3.5)  # prototype contrast weight init

    # ---- Tversky/FT params ----
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=1.2)

    # ---- System & seed ----
    p.add_argument("--in_channels", type=int, default=4)
    p.add_argument("--workers_train", type=int, default=8)
    p.add_argument("--workers_val", type=int, default=4)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--ddp", action="store_true")

    # ---- Meta-hyper (learned weights) ----
    p.add_argument("--use_meta", action="store_true")
    p.add_argument("--meta_lr", type=float, default=1e-3)
    p.add_argument("--init_tau", type=float, default=0.07)
    p.add_argument("--init_lambda_proto", type=float, default=3.5)
    p.add_argument("--init_tversky_w", type=float, default=7.0)
    p.add_argument("--init_imb_w", type=float, default=8.0)

    p.add_argument("--alpha_low", type=float, default=0.05)
    p.add_argument("--alpha_high", type=float, default=0.40)
    p.add_argument("--beta_low", type=float, default=0.60)
    p.add_argument("--beta_high", type=float, default=1.00)
    p.add_argument("--ab_freeze_epochs", type=int, default=10)
    p.add_argument("--ab_kl", type=float, default=1e-3)
    p.add_argument("--proto_warmup_epochs", type=int, default=10)

    # ---- Proto scheduling & alignment ----
    p.add_argument("--proto_floor_start", type=float, default=0.10)
    p.add_argument("--proto_floor_end", type=float, default=0.40)
    p.add_argument("--proto_floor_epochs", type=int, default=60)
    p.add_argument("--proto_align_lambda", type=float, default=5e-3)
    p.add_argument("--proto_target_bias", type=float, default=0.0)
    p.add_argument("--proto_gain_start", type=int, default=30)
    p.add_argument("--proto_gain_span", type=int, default=40)
    p.add_argument("--proto_gain_max", type=float, default=2.0)

    # ---- EMA teacher for TAC ----
    p.add_argument("--new_teacher_is_ema", type=int, default=1, choices=[0, 1])
    p.add_argument("--new_teacher_ema_m", type=float, default=0.99)

    # ---- Replay/Queue & PER ----
    p.add_argument("--mem_size", type=int, default=32000)
    p.add_argument("--p_keep", type=float, default=0.10)

    p.add_argument("--per_enable", type=int, default=1, choices=[0, 1])
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta", type=float, default=0.4)
    p.add_argument("--per_epsilon", type=float, default=1e-6)
    p.add_argument("--per_use_ema", type=int, default=1, choices=[0, 1])
    p.add_argument("--per_ema_m", type=float, default=0.7)

    p.add_argument("--force_first_stage", type=int, default=-1, choices=[-1, 0, 1],
                   help="是否强制视为Stage-1：-1=自动(默认), 1=强制Stage-1 warmup, 0=非Stage-1")
    p.add_argument("--allow_prev_in_4ch", type=int, default=1, choices=[0, 1],
                   help="4通道输入是否允许拼入前序模态：1=允许(默认), 0=不允许")

    # ---- Replay Training (经验回放训练) ----
    p.add_argument("--use_replay", type=int, default=1, choices=[0, 1],
                   help="是否启用replay训练：1=启用(默认), 0=禁用")
    p.add_argument("--replay_ratio", type=float, default=0.3,
                   help="每个batch中replay数据的比例 (默认0.3)")
    p.add_argument("--replay_contrast_weight", type=float, default=1.0,
                   help="replay对比学习损失的权重 (默认1.0)")
    p.add_argument("--replay_stratified", type=int, default=1, choices=[0, 1],
                   help="是否按模态分层抽样：1=是(默认), 0=否")

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

    # helper: just for logging
    def _dataset_modalities_exist(data_path: str):
        mods = []
        for m in ["t1", "t2", "flair", "t1ce"]:
            d = os.path.join(data_path, f"imgs_{m}")
            if os.path.isdir(d):
                mods.append(m)
        return mods

    def _rb_modalities_exist():
        mods = []
        if os.path.isdir("replay_buffer"):
            for fn in os.listdir("replay_buffer"):
                m = re.match(r"^replay_buffer_(.+)\.pth$", fn)
                if m:
                    mods.append(m.group(1))
        return sorted(set(mods))

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

            # Replay/Queue & PER
            mem_size=args.mem_size,
            p_keep=args.p_keep,

            # Core train
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            images_rate=args.images_rate,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            optim_name=args.optim_name,
            lr_scheduler=args.lr_scheduler,
            step_num_lr=args.step_num_lr,

            # Loss weights (static init)
            tversky_w=args.tversky_w,
            imb_w=args.imb_w,
            nce_weight=args.nce_weight,

            # Tversky/FT params
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,

            # System
            in_channels=args.in_channels,
            num_workers_train=args.workers_train,
            num_workers_val=args.workers_val,
            device=device,
            seed=args.seed,

            # Meta-hyper
            use_meta=args.use_meta,
            meta_lr=args.meta_lr,
            init_tau=args.init_tau,
            init_lambda_proto=args.init_lambda_proto,
            init_tversky_w=args.init_tversky_w,
            init_imb_w=args.init_imb_w,
            proto_warmup_epochs=args.proto_warmup_epochs,
            ddp=args.ddp,

            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            beta_low=args.beta_low,
            beta_high=args.beta_high,
            ab_freeze_epochs=args.ab_freeze_epochs,
            ab_kl=args.ab_kl,

            # Proto schedules
            proto_floor_start=args.proto_floor_start,
            proto_floor_end=args.proto_floor_end,
            proto_floor_epochs=args.proto_floor_epochs,
            proto_align_lambda=args.proto_align_lambda,
            proto_target_bias=args.proto_target_bias,
            proto_gain_start=args.proto_gain_start,
            proto_gain_span=args.proto_gain_span,
            proto_gain_max=args.proto_gain_max,

            # EMA teacher for TAC
            new_teacher_is_ema=bool(args.new_teacher_is_ema),
            new_teacher_ema_m=args.new_teacher_ema_m,

            # PER knobs
            per_enable=bool(args.per_enable),
            per_alpha=args.per_alpha,
            per_beta=args.per_beta,
            per_epsilon=args.per_epsilon,
            per_use_ema=bool(args.per_use_ema),
            per_ema_m=args.per_ema_m,

            force_first_stage=(None if args.force_first_stage < 0 else bool(args.force_first_stage)),
            allow_prev_in_4ch=bool(args.allow_prev_in_4ch),

            # Replay Training
            use_replay=bool(args.use_replay),
            replay_ratio=args.replay_ratio,
            replay_contrast_weight=args.replay_contrast_weight,
            replay_stratified=bool(args.replay_stratified),

            # Stage-aware Meta (阶段感知)
            stage_idx=i,                          # 当前阶段索引 (0-based)
            num_stages=len(stages),               # 总阶段数
            meta_equal_init=True,                 # 三个损失权重初始相等
        )

        if is_main:
            logging.info(f"[Stage {i+1}/{len(stages)}] modality={mod}")
            if prev_img_modes:
                logging.info(f" prev_img_modes: {prev_img_modes}")
            if prev_base_dir:
                logging.info(f" prev_base_dir: {prev_base_dir}")
            if args.use_meta:
                logging.info(f" [meta] init_tau={args.init_tau}, lambda_proto={args.init_lambda_proto}, "
                             f"w_tv={args.init_tversky_w}, w_ft={args.init_imb_w}, meta_lr={args.meta_lr}")
            ds_mods = _dataset_modalities_exist(args.data_path)
            rb_mods = _rb_modalities_exist()
            logging.info(f" dataset_modalities: {ds_mods}")
            logging.info(f" replay_buffer_modalities: {rb_mods}")
            logging.info(f" mem_size={args.mem_size} p_keep={args.p_keep} "
                         f"PER(enable={bool(args.per_enable)}, alpha={args.per_alpha}, "
                         f"beta={args.per_beta}, ema={bool(args.per_use_ema)}, ema_m={args.per_ema_m})")
            logging.info(f" replay_training(enable={bool(args.use_replay)}, ratio={args.replay_ratio}, "
                         f"contrast_weight={args.replay_contrast_weight}, stratified={bool(args.replay_stratified)})")

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
