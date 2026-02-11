# -*- coding: utf-8 -*-
"""
Prioritized Experience Replay (PER) for sample selection into replay buffer.

Usage 1 (offline selection at stage end):
    from utils.per_buffer import select_with_per
    sel_names, sel_losses, sel_idx, sel_probs, sel_isw = select_with_per(
        names, losses, keep_ratio=0.10, alpha=0.6, beta=0.4, epsilon=1e-6, seed=42
    )

Usage 2 (online accumulation during training):
    per = PERBuffer(alpha=0.6, beta=0.4, epsilon=1e-6, seed=42)
    per.add_or_update(name, priority=loss, payload={"patient": pid, "mod": mod})
    ...
    idx, batch = per.sample(k)
    per.update(idx, new_priorities)

Notes:
- In supervised segmentation, we can use per-sample supervised loss as "priority".
- alpha controls how strong the prioritization is (0=uniform, 1=full).
- beta controls importance-sampling weight (bias correction).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["PERBuffer", "select_with_per"]


class PERBuffer:
    """
    Lightweight PER buffer with O(N) sampling (sufficient for offline / stage-end selection).
    Stores: names (ids), priorities (>=epsilon), and optional payloads (meta info).
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6, seed: Optional[int] = 42):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        self._names: List[str] = []
        self._priorities: List[float] = []
        self._payloads: List[Optional[Dict[str, Any]]] = []
        self._idx_of_name: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._names)

    def add_or_update(self, name: str, priority: float, payload: Optional[Dict[str, Any]] = None):
        pr = float(abs(priority)) + self.epsilon
        if name in self._idx_of_name:
            i = self._idx_of_name[name]
            self._priorities[i] = pr
            if payload is not None:
                self._payloads[i] = payload
        else:
            i = len(self._names)
            self._idx_of_name[name] = i
            self._names.append(name)
            self._priorities.append(pr)
            self._payloads.append(payload)

    def bulk_build(self, names: Sequence[str], priorities: Sequence[float], payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None):
        assert len(names) == len(priorities), "names/priorities length mismatch"
        self._names = list(map(str, names))
        self._priorities = [float(abs(p)) + self.epsilon for p in priorities]
        if payloads is None:
            self._payloads = [None] * len(self._names)
        else:
            assert len(payloads) == len(names), "payloads length mismatch"
            self._payloads = list(payloads)
        self._idx_of_name = {n: i for i, n in enumerate(self._names)}

    def sample(self, k: int, replace: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        n = len(self)
        if n == 0:
            raise RuntimeError("PERBuffer is empty.")
        k = int(min(max(1, k), n))
        pri = np.asarray(self._priorities, dtype=np.float64)
        probs = pri ** self.alpha
        probs = probs / probs.sum()

        idx = self.rng.choice(n, size=k, replace=replace, p=probs)
        # importance-sampling weights
        with np.errstate(divide="ignore", invalid="ignore"):
            isw = (n * probs[idx]) ** (-self.beta)
            isw = isw / (isw.max() + 1e-12)

        batch = {
            "names": np.array([self._names[i] for i in idx], dtype=object),
            "priorities": pri[idx].astype(np.float64),
            "probs": probs[idx].astype(np.float64),
            "isw": isw.astype(np.float64),
            "payloads": np.array([self._payloads[i] for i in idx], dtype=object),
        }
        return idx, batch

    def update(self, indices: Sequence[int], new_priorities: Sequence[float]):
        assert len(indices) == len(new_priorities), "indices/new_priorities length mismatch"
        for i, p in zip(indices, new_priorities):
            self._priorities[int(i)] = float(abs(p)) + self.epsilon

    def dump(self):
        """Export (names, priorities) for cross-rank aggregation."""
        import numpy as np
        return self._names.copy(), np.asarray(self._priorities, dtype=np.float64).copy()

    # --------- convenience for offline stage-end selection ----------
    def select_topk_by_prob(self, keep_ratio: float, min_keep: int = 1, replace: bool = False) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        n = len(self)
        if n == 0:
            return [], np.array([]), np.array([]), np.array([])
        k = max(min_keep, int(round(float(keep_ratio) * n)))
        k = min(k, n)
        # sample without replacement by probs
        pri = np.asarray(self._priorities, dtype=np.float64)
        probs = pri ** self.alpha
        probs = probs / probs.sum()
        idx = self.rng.choice(n, size=k, replace=replace, p=probs)
        names = [self._names[i] for i in idx]
        return names, pri[idx], probs[idx], idx


def select_with_per(
    names: Sequence[str],
    losses: Sequence[float],
    keep_ratio: float,
    alpha: float = 0.6,
    beta: float = 0.4,
    epsilon: float = 1e-6,
    seed: Optional[int] = 42,
    min_keep: int = 1,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-shot PER selection for saving to replay buffer at stage end.

    Args:
        names:      sample ids (e.g., slice file stems)
        losses:     per-sample supervised loss as priority proxy
        keep_ratio: how many to keep (e.g., 0.10 means keep 10%)
        alpha, beta, epsilon, seed, min_keep: PER hyper-params

    Returns:
        sel_names:  list[str]
        sel_losses: np.ndarray
        sel_idx:    np.ndarray (indices in the original names array)
        sel_probs:  np.ndarray (sampling probabilities used)
        sel_isw:    np.ndarray (importance-sampling weights in [0,1])
    """
    names = list(map(str, names))
    losses = np.asarray(losses, dtype=np.float64)
    assert len(names) == len(losses), "names/losses length mismatch"

    per = PERBuffer(alpha=alpha, beta=beta, epsilon=epsilon, seed=seed)
    per.bulk_build(names, losses)

    n = len(names)
    k = max(min_keep, int(round(float(keep_ratio) * n)))
    k = min(k, n)

    idx, batch = per.sample(k)
    sel_names = batch["names"].tolist()
    sel_losses = batch["priorities"]  # here equals |loss|+eps
    sel_probs = batch["probs"]
    sel_isw = batch["isw"]

    # translate idx from sampled order to original index positions
    # since we built PER in the original order, idx already maps to original order
    sel_idx = idx.astype(np.int64)

    return sel_names, sel_losses, sel_idx, sel_probs, sel_isw
