# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from pathlib import Path

MODS = ["t1", "t2", "flair", "t1ce"]

def load_cases_from_lists(list_paths):
    cases = []
    for lp in list_paths:
        with open(lp, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    cases.append(s)
    seen = set()
    uniq = []
    for c in cases:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def _load_slice(path: Path) -> np.ndarray:
    x = np.load(str(path), allow_pickle=False)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expect 2D slice, got {x.shape} @ {path}")
    return x.astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser("Build BraTS_fusedslice from imgs_{mod}/{case}.npy slices")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--lists", type=str, required=True,
                    help="comma-separated list files (train/val). each line is case_id_sliceid")
    ap.add_argument("--out_dir", type=str, default="", help="default: {data_path}/BraTS_fusedslice")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    ap.add_argument("--fill_missing", type=int, default=0, choices=[0, 1],
                    help="1=00=case")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir) if args.out_dir else (data_path / "BraTS_fusedslice")
    out_dir.mkdir(parents=True, exist_ok=True)

    list_paths = [p.strip() for p in args.lists.split(",") if p.strip()]
    cases = load_cases_from_lists(list_paths)

    dt = np.float16 if args.dtype == "float16" else np.float32

    miss = {m: 0 for m in MODS}
    saved = 0
    skipped = 0

    for idx, case in enumerate(cases):
        outp = out_dir / f"{case}.npy"
        if outp.exists() and not args.overwrite:
            continue

        xs = []
        ok = True
        for m in MODS:
            fp = data_path / f"imgs_{m}" / f"{case}.npy"
            if not fp.exists():
                miss[m] += 1
                if args.fill_missing:
                    xs.append(np.zeros((224, 224), dtype=np.float32))
                else:
                    ok = False
                    break
            else:
                xs.append(_load_slice(fp))

        if not ok:
            skipped += 1
            continue

        for x in xs:
            if x.shape != (224, 224):
                raise ValueError(f"Shape mismatch for case={case}: got {x.shape}, want (224,224)")

        fused = np.stack(xs, axis=0).astype(dt, copy=False)  # [4,224,224]
        np.save(str(outp), fused, allow_pickle=False)
        saved += 1


if __name__ == "__main__":
    main()
