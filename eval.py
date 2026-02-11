#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval.py - 评估脚本 for PMD (Prototype-guided Multi-modal DIL Segmentation)

支持:
- 多模态评估 (t1, t2, flair, t1ce 及其组合)
- 15 种模态缺失场景
- TTA (Test-Time Augmentation)
- 批量评估多个 stage 的模型
- 详细指标: Dice, Sensitivity, PPV per region (WT/TC/ET)

Usage:
    python eval.py --model_path res-t1ce/model_CPH_best.pth --data_path /path/to/data --test_list test.list
    python eval.py --model_path res-t1ce/model_CPH_best.pth --eval_all_codes --use_TTA
"""

import os
import gc
import copy
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nets.cph import CPH
from dataloader.dataset import BaseDataSets
from utils.metrics import dice as dice_single, batch_dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 模态顺序: t1, t2, flair, t1ce
MODS = ["t1", "t2", "flair", "t1ce"]
REGIONS = ["WT", "TC", "ET"]

# 15 种模态组合 (用于完整评估)
ALL_MODAL_CODES = [
    [1, 1, 1, 1],  # 0: all modalities
    [0, 1, 1, 1],  # 1: missing t1
    [1, 0, 1, 1],  # 2: missing t2
    [1, 1, 0, 1],  # 3: missing flair
    [1, 1, 1, 0],  # 4: missing t1ce
    [0, 0, 1, 1],  # 5: only flair + t1ce
    [0, 1, 0, 1],  # 6: only t2 + t1ce
    [0, 1, 1, 0],  # 7: only t2 + flair
    [1, 0, 0, 1],  # 8: only t1 + t1ce
    [1, 0, 1, 0],  # 9: only t1 + flair
    [1, 1, 0, 0],  # 10: only t1 + t2
    [1, 0, 0, 0],  # 11: only t1
    [0, 1, 0, 0],  # 12: only t2
    [0, 0, 1, 0],  # 13: only flair
    [0, 0, 0, 1],  # 14: only t1ce
]


def parse_args():
    parser = argparse.ArgumentParser(description='PMD Evaluation Script')
    
    # 模型和数据路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (e.g., res-t1ce/model_CPH_best.pth)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--test_list', type=str, default='test.list',
                        help='Test list filename')
    parser.add_argument('--fused_dir', type=str, default='BraTS_fusedslice',
                        help='Directory name for 4-channel fused images')
    
    # 评估设置
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--in_channels', type=int, default=4,
                        help='Number of input channels')
    
    # 模态设置
    parser.add_argument('--modal_code', type=str, default='1,1,1,1',
                        help='Modal availability code (comma-separated, e.g., "1,1,1,0" for missing t1ce)')
    parser.add_argument('--eval_all_codes', action='store_true',
                        help='Evaluate all 15 modal codes')
    
    # TTA 设置
    parser.add_argument('--use_TTA', action='store_true',
                        help='Use Test-Time Augmentation (horizontal/vertical flip)')
    parser.add_argument('--tta_modes', type=str, default='hflip,vflip,hv',
                        help='TTA modes: hflip, vflip, hv (comma-separated)')
    
    # 输出设置
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as model_path)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    
    # GPU 设置
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


class EvalDataset(Dataset):
    """
    评估用数据集: 加载 4 通道融合图像并应用模态掩码
    """
    def __init__(
        self,
        data_dir: str,
        list_name: str,
        fused_dir: str = 'BraTS_fusedslice',
        modal_code: List[int] = None,
        fallback_mods: List[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.fused_dir = self.data_dir / fused_dir
        self.mask_dir = self.data_dir / 'masks_all'
        if not self.mask_dir.exists():
            self.mask_dir = self.data_dir / 'masks'
        
        # 模态掩码
        self.modal_code = np.array(modal_code if modal_code else [1, 1, 1, 1], dtype=np.float32)
        
        # 单模态目录 (fallback)
        self.mod_dirs = {m: self.data_dir / f'imgs_{m}' for m in MODS}
        
        # 加载样本列表
        self.sample_list = []
        list_path = self.data_dir / list_name
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.sample_list.append(line)
        
        logging.info(f"[Eval] Loaded {len(self.sample_list)} samples from {list_path}")
        logging.info(f"[Eval] Modal code: {self.modal_code.tolist()} ({self._code_to_str()})")
    
    def _code_to_str(self) -> str:
        """将模态码转换为可读字符串"""
        present = [MODS[i] for i in range(4) if self.modal_code[i] > 0]
        return '+'.join(present) if present else 'none'
    
    def __len__(self):
        return len(self.sample_list)
    
    def _load_fused(self, sid: str) -> Optional[np.ndarray]:
        """加载 4 通道融合图像"""
        path = self.fused_dir / f'{sid}.npy'
        if path.exists():
            arr = np.load(str(path)).astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 4:
                return arr
            if arr.ndim == 3 and arr.shape[-1] == 4:
                return np.transpose(arr, (2, 0, 1))
        return None
    
    def _load_single_mods(self, sid: str) -> np.ndarray:
        """从单模态目录加载并堆叠"""
        H, W = None, None
        channels = []
        
        for i, mod in enumerate(MODS):
            path = self.mod_dirs[mod] / f'{sid}.npy'
            if path.exists():
                arr = np.load(str(path)).astype(np.float32)
                if arr.ndim == 2:
                    pass
                elif arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                else:
                    arr = arr.squeeze()
                
                if H is None:
                    H, W = arr.shape
                channels.append(arr)
            else:
                channels.append(None)
        
        if H is None:
            raise FileNotFoundError(f"No modality found for {sid}")
        
        out = np.zeros((4, H, W), dtype=np.float32)
        for i, ch in enumerate(channels):
            if ch is not None:
                out[i] = ch
        return out
    
    def __getitem__(self, idx):
        sid = self.sample_list[idx]
        
        # 加载图像
        img4 = self._load_fused(sid)
        if img4 is None:
            img4 = self._load_single_mods(sid)
        
        # 应用模态掩码
        mask_4d = self.modal_code.reshape(4, 1, 1)
        img4 = img4 * mask_4d
        
        # 加载 ground truth mask
        mask_path = self.mask_dir / f'{sid}.npy'
        gt_mask = np.load(str(mask_path)).astype(np.float32)
        if gt_mask.ndim == 2:
            gt_mask = np.expand_dims(gt_mask, axis=0)
        
        return {
            'image': torch.from_numpy(img4),
            'mask': torch.from_numpy(gt_mask),
            'idx': sid,
            'modal_code': torch.from_numpy(self.modal_code.copy()),
        }


def apply_tta(model: nn.Module, x: torch.Tensor, tta_modes: List[str]) -> torch.Tensor:
    """
    应用 Test-Time Augmentation
    
    Args:
        model: 模型
        x: 输入图像 [B, C, H, W]
        tta_modes: TTA 模式列表
    
    Returns:
        平均后的预测结果
    """
    outputs = []
    
    # 原始预测
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    outputs.append(torch.sigmoid(out))
    
    # 水平翻转
    if 'hflip' in tta_modes:
        x_flip = torch.flip(x, dims=[3])
        out_flip = model(x_flip)
        if isinstance(out_flip, tuple):
            out_flip = out_flip[0]
        outputs.append(torch.flip(torch.sigmoid(out_flip), dims=[3]))
    
    # 垂直翻转
    if 'vflip' in tta_modes:
        x_flip = torch.flip(x, dims=[2])
        out_flip = model(x_flip)
        if isinstance(out_flip, tuple):
            out_flip = out_flip[0]
        outputs.append(torch.flip(torch.sigmoid(out_flip), dims=[2]))
    
    # 水平 + 垂直翻转
    if 'hv' in tta_modes or 'hvflip' in tta_modes:
        x_flip = torch.flip(x, dims=[2, 3])
        out_flip = model(x_flip)
        if isinstance(out_flip, tuple):
            out_flip = out_flip[0]
        outputs.append(torch.flip(torch.sigmoid(out_flip), dims=[2, 3]))
    
    # 平均
    return torch.stack(outputs, dim=0).mean(dim=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    计算单个样本的分割指标
    
    Args:
        pred: 预测 [3, H, W] (WT, TC, ET)
        target: 标签 [3, H, W]
    
    Returns:
        包含 Dice, Sensitivity, PPV 的字典
    """
    metrics = {}
    
    for i, region in enumerate(REGIONS):
        p = pred[i].astype(np.uint8)
        t = target[i].astype(np.uint8)
        
        dice_val, ppv_val, sen_val = dice_single(p, t)
        
        metrics[f'{region}_dice'] = dice_val
        metrics[f'{region}_sen'] = sen_val
        metrics[f'{region}_ppv'] = ppv_val
    
    # 平均 Dice
    metrics['avg_dice'] = np.mean([metrics[f'{r}_dice'] for r in REGIONS])
    
    return metrics


def evaluate_single_code(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = False,
    tta_modes: List[str] = None,
    save_predictions: bool = False,
    output_dir: Path = None,
) -> Dict[str, List[float]]:
    """
    对单个模态组合进行评估
    """
    model.eval()
    
    all_metrics = {f'{r}_{m}': [] for r in REGIONS for m in ['dice', 'sen', 'ppv']}
    all_metrics['avg_dice'] = []
    
    sample_results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].numpy()
            indices = batch['idx']
            
            # 前向传播
            if use_tta and tta_modes:
                probs = apply_tta(model, images, tta_modes)
            else:
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.sigmoid(logits)
            
            # 转换为 numpy
            probs_np = probs.cpu().numpy()
            
            # 批量处理
            B = images.size(0)
            for b in range(B):
                pred = (probs_np[b] > threshold).astype(np.float32)
                target = masks[b]
                
                # 确保形状正确
                if pred.shape[0] != 3:
                    logging.warning(f"Unexpected pred shape: {pred.shape}")
                    continue
                if target.shape[0] != 3:
                    # 可能需要转换 mask 格式
                    if target.ndim == 2:
                        # 单通道 mask，需要转换为 3 通道 (WT/TC/ET)
                        # 假设: 0=background, 1=WT, 2=TC, 3=ET
                        # 这里需要根据实际数据格式调整
                        pass
                
                # 计算指标
                metrics = compute_metrics(pred, target)
                
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                sample_results.append({
                    'idx': indices[b],
                    **metrics
                })
                
                # 保存预测
                if save_predictions and output_dir:
                    pred_path = output_dir / 'predictions' / f'{indices[b]}.npy'
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(pred_path), pred)
    
    return all_metrics, sample_results


def evaluate(args):
    """主评估函数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 加载模型
    logging.info(f"Loading model from {args.model_path}")
    model = CPH(n_classes=3, in_channels=args.in_channels).to(device)
    
    state_dict = torch.load(args.model_path, map_location=device)
    # 处理可能的 DDP wrapper
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logging.info(f"Model loaded successfully")
    
    # TTA 设置
    tta_modes = args.tta_modes.split(',') if args.use_TTA else []
    
    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要评估的模态组合
    if args.eval_all_codes:
        codes_to_eval = ALL_MODAL_CODES
        logging.info(f"Evaluating all {len(codes_to_eval)} modal codes")
    else:
        code = [int(x) for x in args.modal_code.split(',')]
        codes_to_eval = [code]
    
    # 存储所有结果
    all_results = []
    
    gc.disable()
    
    for code_idx, modal_code in enumerate(codes_to_eval):
        code_str = ''.join(map(str, modal_code))
        present_mods = [MODS[i] for i in range(4) if modal_code[i] > 0]
        logging.info(f"\n{'='*60}")
        logging.info(f"[{code_idx+1}/{len(codes_to_eval)}] Evaluating modal code: {modal_code}")
        logging.info(f"Present modalities: {present_mods}")
        logging.info(f"{'='*60}")
        
        # 创建数据集
        dataset = EvalDataset(
            data_dir=args.data_path,
            list_name=args.test_list,
            fused_dir=args.fused_dir,
            modal_code=modal_code,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        # 评估
        metrics, sample_results = evaluate_single_code(
            model=model,
            data_loader=dataloader,
            device=device,
            threshold=args.threshold,
            use_tta=args.use_TTA,
            tta_modes=tta_modes,
            save_predictions=args.save_predictions,
            output_dir=output_dir / f'code_{code_str}' if args.save_predictions else None,
        )
        
        # 汇总结果
        result = {
            'modal_code': modal_code,
            'code_str': code_str,
            'present_mods': present_mods,
            'num_samples': len(dataset),
        }
        
        for region in REGIONS:
            result[f'{region}_dice'] = np.mean(metrics[f'{region}_dice'])
            result[f'{region}_sen'] = np.mean(metrics[f'{region}_sen'])
            result[f'{region}_ppv'] = np.mean(metrics[f'{region}_ppv'])
        
        result['avg_dice'] = np.mean(metrics['avg_dice'])
        
        all_results.append(result)
        
        # 打印结果
        logging.info(f"\nResults for code {code_str}:")
        logging.info(f"  WT Dice: {result['WT_dice']:.4f} | Sen: {result['WT_sen']:.4f} | PPV: {result['WT_ppv']:.4f}")
        logging.info(f"  TC Dice: {result['TC_dice']:.4f} | Sen: {result['TC_sen']:.4f} | PPV: {result['TC_ppv']:.4f}")
        logging.info(f"  ET Dice: {result['ET_dice']:.4f} | Sen: {result['ET_sen']:.4f} | PPV: {result['ET_ppv']:.4f}")
        logging.info(f"  Avg Dice: {result['avg_dice']:.4f}")
    
    gc.enable()
    
    # 汇总所有结果
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY OF ALL MODAL CODES")
    logging.info(f"{'='*80}")
    
    print("\n" + "-"*100)
    print(f"{'Code':<15} {'Modalities':<20} {'WT Dice':<12} {'TC Dice':<12} {'ET Dice':<12} {'Avg Dice':<12}")
    print("-"*100)
    
    for r in all_results:
        mods_str = '+'.join(r['present_mods']) if r['present_mods'] else 'none'
        print(f"{r['code_str']:<15} {mods_str:<20} {r['WT_dice']:.4f}       {r['TC_dice']:.4f}       {r['ET_dice']:.4f}       {r['avg_dice']:.4f}")
    
    print("-"*100)
    
    # 计算平均
    if len(all_results) > 1:
        avg_wt = np.mean([r['WT_dice'] for r in all_results])
        avg_tc = np.mean([r['TC_dice'] for r in all_results])
        avg_et = np.mean([r['ET_dice'] for r in all_results])
        avg_all = np.mean([r['avg_dice'] for r in all_results])
        
        print(f"{'MEAN':<15} {'all codes':<20} {avg_wt:.4f}       {avg_tc:.4f}       {avg_et:.4f}       {avg_all:.4f}")
        print("-"*100)
    
    # 保存结果到文件
    result_file = output_dir / 'eval_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Test List: {args.test_list}\n")
        f.write(f"TTA: {args.use_TTA}\n\n")
        
        f.write("-"*100 + "\n")
        f.write(f"{'Code':<15} {'Modalities':<20} {'WT Dice':<12} {'TC Dice':<12} {'ET Dice':<12} {'Avg Dice':<12}\n")
        f.write("-"*100 + "\n")
        
        for r in all_results:
            mods_str = '+'.join(r['present_mods']) if r['present_mods'] else 'none'
            f.write(f"{r['code_str']:<15} {mods_str:<20} {r['WT_dice']:.4f}       {r['TC_dice']:.4f}       {r['ET_dice']:.4f}       {r['avg_dice']:.4f}\n")
        
        if len(all_results) > 1:
            f.write("-"*100 + "\n")
            f.write(f"{'MEAN':<15} {'all codes':<20} {avg_wt:.4f}       {avg_tc:.4f}       {avg_et:.4f}       {avg_all:.4f}\n")
    
    logging.info(f"\nResults saved to {result_file}")
    
    return all_results


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
