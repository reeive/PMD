import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch
from torch.utils.data import Dataset, Sampler

MODS = ["t1", "t2", "flair", "t1ce"]


def _load_replay(modality: str, device: str = "cpu") -> Optional[dict]:
    path = os.path.join("replay_buffer", f"replay_buffer_{modality}.pth")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt.setdefault("patients", ["unk"] * ckpt["images"].shape[0])
    ckpt.setdefault("modalities", [modality] * ckpt["images"].shape[0])
    return ckpt


def _save_replay(
    modality: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    losses: torch.Tensor,
    patients: List[str],
    modalities: List[str],
):
    os.makedirs("replay_buffer", exist_ok=True)
    path = os.path.join("replay_buffer", f"replay_buffer_{modality}.pth")
    torch.save(
        {
            "images": images.cpu(),
            "masks": masks.cpu(),
            "losses": losses.cpu(),
            "patients": patients,
            "modalities": modalities,
        },
        path,
    )
    logging.info(f"[Replay] Saved {images.shape[0]} samples to {path}")


def _load_all_replay_buffers(prev_modalities: List[str], device: str = "cpu") -> Dict[str, dict]:
    """
    
    Returns:
        Dict[modality -> replay_data]
    """
    replay_dict = {}
    for mod in prev_modalities:
        rb = _load_replay(mod, device=device)
        if rb is not None and rb["images"].shape[0] > 0:
            replay_dict[mod] = rb
            logging.info(f"[Replay] Loaded {rb['images'].shape[0]} samples from {mod}")
    return replay_dict


class ReplayDataset(Dataset):
    """
    """
    def __init__(
        self,
        replay_dict: Dict[str, dict],
        fused_dir: Optional[Path] = None,
        allowed_mask: Optional[torch.Tensor] = None,
        dtype: str = "float16",
    ):
        """
        Args:
            replay_dict: {modality: {images, masks, losses, patients, modalities}}
            fused_dir: 4ch fused slice  (4)
            allowed_mask: [4]
        """
        self.replay_dict = replay_dict
        self.fused_dir = Path(fused_dir) if fused_dir else None
        self.allowed = allowed_mask if allowed_mask is not None else torch.ones(4)
        self.dt = np.float16 if dtype == "float16" else np.float32
        
        self.index_map: List[Tuple[str, int]] = []
        self.mod_to_indices: Dict[str, List[int]] = {}
        
        global_idx = 0
        for mod, data in replay_dict.items():
            n_samples = data["images"].shape[0]
            self.mod_to_indices[mod] = []
            for local_idx in range(n_samples):
                self.index_map.append((mod, local_idx))
                self.mod_to_indices[mod].append(global_idx)
                global_idx += 1
        
        self.total_samples = len(self.index_map)
        logging.info(f"[ReplayDataset] Total {self.total_samples} samples from {list(replay_dict.keys())}")
    
    def __len__(self):
        return self.total_samples
    
    def _patient_from_sid(self, sid: str) -> str:
        sid = str(sid)
        return sid.rsplit("_", 1)[0]
    
    def __getitem__(self, idx: int):
        mod, local_idx = self.index_map[idx]
        data = self.replay_dict[mod]
        
        img = data["images"][local_idx]
        mask = data["masks"][local_idx]  # [3, H, W]
        loss = data["losses"][local_idx] if "losses" in data else 0.0
        patient = data["patients"][local_idx] if "patients" in data else "unk"
        
        img = img.float()
        mask = mask.float()
        
        if img.dim() == 3 and img.shape[0] == 1:
            H, W = img.shape[1], img.shape[2]
            img4 = torch.zeros(4, H, W, dtype=torch.float32)
            mod_idx = MODS.index(mod) if mod in MODS else 0
            img4[mod_idx] = img[0]
            img = img4
        elif img.dim() == 3 and img.shape[0] == 4:
            pass
        else:
            if img.dim() == 2:
                H, W = img.shape
                img4 = torch.zeros(4, H, W, dtype=torch.float32)
                mod_idx = MODS.index(mod) if mod in MODS else 0
                img4[mod_idx] = img
                img = img4
        
        present = torch.zeros(4, dtype=torch.float32)
        mod_idx = MODS.index(mod) if mod in MODS else 0
        present[mod_idx] = 1.0
        present = present * self.allowed.float()
        img = img * present.view(4, 1, 1)
        
        return {
            "image4_full": img,
            "present_full": present,
            "mask": mask,
            "idx": f"replay_{mod}_{local_idx}",
            "patient": patient,
            "modality": mod,
            "loss": float(loss),
            "is_replay": True,
        }
    
    def get_modality_indices(self, modality: str) -> List[int]:
        return self.mod_to_indices.get(modality, [])
    
    def get_all_modalities(self) -> List[str]:
        return list(self.mod_to_indices.keys())


class StratifiedReplaySampler(Sampler):
    """
    """
    def __init__(
        self,
        replay_dataset: ReplayDataset,
        samples_per_epoch: int,
        stratify_by_mod: bool = True,
        seed: int = 42,
    ):
        self.dataset = replay_dataset
        self.samples_per_epoch = samples_per_epoch
        self.stratify_by_mod = stratify_by_mod
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.rng = np.random.default_rng(self.epoch + 42)
    
    def __iter__(self):
        if not self.stratify_by_mod:
            indices = self.rng.choice(len(self.dataset), size=self.samples_per_epoch, replace=True)
            return iter(indices.tolist())
        
        mods = self.dataset.get_all_modalities()
        if len(mods) == 0:
            return iter([])
        
        samples_per_mod = max(1, self.samples_per_epoch // len(mods))
        indices = []
        
        for mod in mods:
            mod_indices = self.dataset.get_modality_indices(mod)
            if len(mod_indices) == 0:
                continue
            sampled = self.rng.choice(mod_indices, size=samples_per_mod, replace=True)
            indices.extend(sampled.tolist())
        
        if len(indices) < self.samples_per_epoch:
            all_indices = list(range(len(self.dataset)))
            extra = self.rng.choice(all_indices, size=self.samples_per_epoch - len(indices), replace=True)
            indices.extend(extra.tolist())
        
        self.rng.shuffle(indices)
        return iter(indices[:self.samples_per_epoch])
    
    def __len__(self):
        return self.samples_per_epoch


class MixedBatchSampler:
    """
    
    - (1 - current_ratio)  replay  ()
    """
    def __init__(
        self,
        current_dataset_size: int,
        replay_dataset: Optional[ReplayDataset],
        batch_size: int,
        current_ratio: float = 0.7,
        seed: int = 42,
    ):
        self.current_size = current_dataset_size
        self.replay_ds = replay_dataset
        self.batch_size = batch_size
        self.current_ratio = current_ratio
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        
        self.current_per_batch = max(1, int(batch_size * current_ratio))
        self.replay_per_batch = batch_size - self.current_per_batch
        
        if replay_dataset is None or len(replay_dataset) == 0:
            self.current_per_batch = batch_size
            self.replay_per_batch = 0
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.rng = np.random.default_rng(self.epoch + 42)
    
    def sample_replay_batch(self, size: int) -> List[int]:
        if self.replay_ds is None or len(self.replay_ds) == 0 or size == 0:
            return []
        
        mods = self.replay_ds.get_all_modalities()
        if len(mods) == 0:
            return []
        
        per_mod = max(1, size // len(mods))
        indices = []
        for mod in mods:
            mod_indices = self.replay_ds.get_modality_indices(mod)
            if len(mod_indices) == 0:
                continue
            sampled = self.rng.choice(mod_indices, size=min(per_mod, len(mod_indices)), replace=True)
            indices.extend(sampled.tolist())
        
        if len(indices) < size:
            all_idx = list(range(len(self.replay_ds)))
            extra = self.rng.choice(all_idx, size=size - len(indices), replace=True)
            indices.extend(extra.tolist())
        
        self.rng.shuffle(indices)
        return indices[:size]
    
    def sample_current_batch(self, size: int) -> List[int]:
        return self.rng.choice(self.current_size, size=size, replace=False).tolist()
    
    def get_batch_config(self) -> Tuple[int, int]:
        """ (current_per_batch, replay_per_batch)"""
        return self.current_per_batch, self.replay_per_batch
