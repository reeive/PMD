# From Semantic Drift to Objective Miscalibration: Domain Incremental Brain Tumor Segmentation with Missing Modality

## Overview

PMD formulates missing-modality brain tumor segmentation as a Domain Incremental Learning (DIL) problem, where MRI modalities (T1, T2, FLAIR, T1CE) are introduced sequentially. PMD addresses two key challenges:

- **Semantic Drift (SD)**: Stage-wise representation shifts make previously learned semantic references progressively unreliable, causing order-sensitivity across different incremental orders.
- **Objective Miscalibration (OM)**: Fixed loss weights fail to account for modality-dependent error characteristics, leading to unstable optimization across stages.

## Installation

```bash
# Clone the repository
git clone https://github.com/reeive/PMD.git
cd PMD

# Install dependencies
pip install torch torchvision
pip install numpy scipy scikit-learn tqdm
```

**Requirements**: Python >= 3.8, PyTorch >= 1.12, CUDA >= 11.3

## Data Preparation

### BraTS 2019

1. Download BraTS 2019 from [CBICA](https://www.med.upenn.edu/cbica/brats2019/data.html)
2. Extract 2D slices with the provided preprocessing script
3. Organize as:

```
data/
├── BraTS_fusedslice/        # Fused 4-channel slices
│   ├── patient001_slice080.npy
│   └── ...
├── masks_all/               # Segmentation masks
│   ├── patient001_slice080.npy
│   └── ...
├── imgs_t1/                 # Single-modality slices (optional fallback)
├── imgs_t2/
├── imgs_flair/
├── imgs_t1ce/
└── lists/
    ├── train.list
    └── val.list
```

### FeTS 2022 and MU-Glioma-Post

Used for OOD evaluation. Apply the same slicing and cropping protocol as BraTS 2019 to obtain 224x224 2D slices.

## Training

### Default (clinical order: T1 -> T2 -> FLAIR -> T1CE)

```bash
python train.py \
    --data_path /path/to/data \
    --out_root results/clinical_order \
    --stages t1,t2,flair,t1ce \
    --use_meta \
    --gpus 0
```

### Feature-driven order

```bash
python train.py \
    --data_path /path/to/data \
    --out_root results/feature_order \
    --stages flair,t1ce,t1,t2 \
    --use_meta \
    --gpus 0
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 train.py \
    --data_path /path/to/data \
    --out_root results/clinical_order \
    --stages t1,t2,flair,t1ce \
    --use_meta \
    --ddp
```

## Acknowledgments

This work extends our prior MICCAI 2025 work [ReHyDIL](https://github.com/reeive/PMD) with three improvements: prototype-based Tversky-Aware Contrastive loss (pTAC) with PRM, an online Meta Controller for stage-adaptive loss reweighting, and expanded evaluation quantifying order sensitivity and OOD generalization.
