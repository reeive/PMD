#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_stages.py - 多阶段模型评估脚本 for PMD

评估 Domain Incremental Learning 各阶段的模型性能，
包括:
- 每个阶段在当前模态和所有之前模态上的性能
- 遗忘分析 (Forgetting Analysis)
- 患者级指标聚合
- 完整的 15 种模态组合评估

Usage:
    python eval_stages.py --out_root ./outputs --data_path /path/to/data --test_list test.list
    python eval_stages.py --out_root ./outputs --stages t1,t2,flair,t1ce --eval_forgetting
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.cph import CPH
from eval import EvalDataset, evaluate_single_code, ALL_MODAL_CODES, MODS, REGIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-stage PMD Evaluation')
    
    # 路径设置
    parser.add_argument('--out_root', type=str, required=True,
                        help='Root directory containing stage outputs (e.g., res-t1, res-t2, ...)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--test_list', type=str, default='test.list',
                        help='Test list filename')
    parser.add_argument('--fused_dir', type=str, default='imgs_4ch',
                        help='Directory name for 4-channel fused images')
    
    # 阶段设置
    parser.add_argument('--stages', type=str, default='t1,t2,flair,t1ce',
                        help='Comma-separated list of stages to evaluate')
    parser.add_argument('--model_name', type=str, default='model_CPH_best.pth',
                        help='Model checkpoint filename')
    
    # 评估设置
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--use_TTA', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    # 分析选项
    parser.add_argument('--eval_all_codes', action='store_true',
                        help='Evaluate all 15 modal combinations')
    parser.add_argument('--eval_forgetting', action='store_true',
                        help='Compute forgetting metrics across stages')
    parser.add_argument('--eval_backward_transfer', action='store_true',
                        help='Evaluate backward transfer (how later stages improve earlier modalities)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def get_stage_modal_code(stage: str) -> List[int]:
    """获取某阶段对应的模态码"""
    stage_idx = MODS.index(stage)
    code = [0, 0, 0, 0]
    for i in range(stage_idx + 1):
        code[i] = 1
    return code


def load_model(model_path: Path, device: torch.device, in_channels: int = 4) -> nn.Module:
    """加载模型"""
    model = CPH(n_classes=3, in_channels=in_channels).to(device)
    
    state_dict = torch.load(str(model_path), map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 尝试加载，忽略不匹配的键
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys: {missing[:5]}...")
    
    model.eval()
    return model


def evaluate_stage(
    model: nn.Module,
    data_path: str,
    test_list: str,
    fused_dir: str,
    modal_code: List[int],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    threshold: float = 0.5,
    use_tta: bool = False,
    tta_modes: List[str] = None,
) -> Dict[str, float]:
    """评估单个阶段在指定模态组合上的性能"""
    
    dataset = EvalDataset(
        data_dir=data_path,
        list_name=test_list,
        fused_dir=fused_dir,
        modal_code=modal_code,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    metrics, _ = evaluate_single_code(
        model=model,
        data_loader=dataloader,
        device=device,
        threshold=threshold,
        use_tta=use_tta,
        tta_modes=tta_modes or [],
    )
    
    # 汇总
    result = {
        'num_samples': len(dataset),
    }
    for region in REGIONS:
        result[f'{region}_dice'] = np.mean(metrics[f'{region}_dice'])
        result[f'{region}_sen'] = np.mean(metrics[f'{region}_sen'])
        result[f'{region}_ppv'] = np.mean(metrics[f'{region}_ppv'])
    result['avg_dice'] = np.mean(metrics['avg_dice'])
    
    return result


def compute_forgetting(stage_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    计算遗忘指标
    
    Forgetting for modality M at stage S = 
        max(performance on M at stages 1..S-1) - performance on M at stage S
    
    正值表示遗忘，负值表示改进（backward transfer）
    """
    forgetting = {}
    stages = list(stage_results.keys())
    
    for i, stage in enumerate(stages):
        if i == 0:
            continue  # 第一阶段没有遗忘
        
        forgetting[stage] = {}
        
        # 对之前的每个模态计算遗忘
        for j in range(i):
            prev_mod = stages[j]
            prev_code = get_stage_modal_code(prev_mod)
            code_str = ''.join(map(str, prev_code))
            
            # 找到该模态在之前各阶段的最佳性能
            best_prev_perf = 0.0
            for k in range(j, i):
                prev_stage = stages[k]
                if code_str in stage_results[prev_stage]:
                    perf = stage_results[prev_stage][code_str]['avg_dice']
                    best_prev_perf = max(best_prev_perf, perf)
            
            # 当前阶段在该模态上的性能
            if code_str in stage_results[stage]:
                current_perf = stage_results[stage][code_str]['avg_dice']
                forgetting[stage][prev_mod] = best_prev_perf - current_perf
    
    return forgetting


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    stages = args.stages.split(',')
    out_root = Path(args.out_root)
    
    output_dir = Path(args.output_dir) if args.output_dir else out_root / 'eval_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Evaluating stages: {stages}")
    logging.info(f"Output directory: {output_dir}")
    
    # 存储所有结果
    all_stage_results = {}
    
    # 评估每个阶段
    for stage_idx, stage in enumerate(stages):
        stage_dir = out_root / f'res-{stage}'
        model_path = stage_dir / args.model_name
        
        if not model_path.exists():
            logging.warning(f"Model not found: {model_path}, skipping stage {stage}")
            continue
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage}")
        logging.info(f"Model: {model_path}")
        logging.info(f"{'='*80}")
        
        model = load_model(model_path, device, args.in_channels)
        
        stage_results = {}
        
        if args.eval_all_codes:
            # 评估所有 15 种组合
            codes_to_eval = ALL_MODAL_CODES
        else:
            # 只评估到当前阶段的模态组合
            codes_to_eval = []
            for i in range(stage_idx + 1):
                code = get_stage_modal_code(stages[i])
                codes_to_eval.append(code)
            # 也添加完整模态
            codes_to_eval.append([1, 1, 1, 1])
        
        for modal_code in codes_to_eval:
            code_str = ''.join(map(str, modal_code))
            present_mods = [MODS[i] for i in range(4) if modal_code[i] > 0]
            
            logging.info(f"\n  Evaluating code: {modal_code} ({'+'.join(present_mods)})")
            
            try:
                result = evaluate_stage(
                    model=model,
                    data_path=args.data_path,
                    test_list=args.test_list,
                    fused_dir=args.fused_dir,
                    modal_code=modal_code,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    threshold=args.threshold,
                    use_tta=args.use_TTA,
                    tta_modes=['hflip', 'vflip'] if args.use_TTA else None,
                )
                
                stage_results[code_str] = result
                
                logging.info(f"    WT: {result['WT_dice']:.4f} | TC: {result['TC_dice']:.4f} | "
                           f"ET: {result['ET_dice']:.4f} | Avg: {result['avg_dice']:.4f}")
                
            except Exception as e:
                logging.error(f"    Error: {e}")
                continue
        
        all_stage_results[stage] = stage_results
        
        # 释放模型
        del model
        torch.cuda.empty_cache()
    
    # 汇总结果
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY")
    logging.info(f"{'='*80}")
    
    # 打印表格
    print("\n" + "="*120)
    print("Stage-wise Performance (Avg Dice on each modal combination)")
    print("="*120)
    
    # 表头
    header = f"{'Stage':<12}"
    for code in ALL_MODAL_CODES if args.eval_all_codes else [[1,1,1,1]]:
        code_str = ''.join(map(str, code))
        header += f" {code_str:<8}"
    print(header)
    print("-"*120)
    
    # 数据行
    for stage in stages:
        if stage not in all_stage_results:
            continue
        row = f"{stage:<12}"
        for code in ALL_MODAL_CODES if args.eval_all_codes else [[1,1,1,1]]:
            code_str = ''.join(map(str, code))
            if code_str in all_stage_results[stage]:
                val = all_stage_results[stage][code_str]['avg_dice']
                row += f" {val:.4f}  "
            else:
                row += f" {'--':<8}"
        print(row)
    
    print("="*120)
    
    # 遗忘分析
    if args.eval_forgetting and len(stages) > 1:
        logging.info("\n" + "="*80)
        logging.info("FORGETTING ANALYSIS")
        logging.info("="*80)
        
        forgetting = compute_forgetting(all_stage_results)
        
        print("\nForgetting (positive = performance drop, negative = improvement):")
        print("-"*60)
        
        for stage, forg in forgetting.items():
            print(f"\nAt stage {stage}:")
            for mod, val in forg.items():
                sign = "↓" if val > 0 else "↑"
                print(f"  {mod}: {val:+.4f} {sign}")
        
        # 平均遗忘
        avg_forg = np.mean([v for f in forgetting.values() for v in f.values()])
        print(f"\nAverage Forgetting: {avg_forg:+.4f}")
    
    # 保存结果到 JSON
    result_file = output_dir / 'stage_results.json'
    with open(result_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'stage_results': all_stage_results,
        }, f, indent=2, default=float)
    
    logging.info(f"\nResults saved to {result_file}")
    
    # 保存可读的表格
    txt_file = output_dir / 'stage_results.txt'
    with open(txt_file, 'w') as f:
        f.write("PMD Multi-Stage Evaluation Results\n")
        f.write("="*80 + "\n\n")
        
        for stage in stages:
            if stage not in all_stage_results:
                continue
            
            f.write(f"\nStage: {stage}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Code':<15} {'WT Dice':<12} {'TC Dice':<12} {'ET Dice':<12} {'Avg Dice':<12}\n")
            f.write("-"*60 + "\n")
            
            for code_str, result in all_stage_results[stage].items():
                f.write(f"{code_str:<15} {result['WT_dice']:.4f}       "
                       f"{result['TC_dice']:.4f}       {result['ET_dice']:.4f}       "
                       f"{result['avg_dice']:.4f}\n")
    
    logging.info(f"Results saved to {txt_file}")


if __name__ == '__main__':
    main()
