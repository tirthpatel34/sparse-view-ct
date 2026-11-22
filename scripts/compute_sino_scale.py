"""
Find the constant that maps your sinograms into [-1, 1].

    python scripts/compute_sino_scale.py --config configs/ct.yaml

Why this exists
---------------
Neural networks are usually trained on data in roughly [-1, 1]. A raw
sinogram is nothing like that:
each value is a line integral, i.e. a SUM of pixel values along a ray, so it
can be tens or hundreds.

So we divide every sinogram by one constant. It must be the SAME constant for
every sample and every split -- a per-image scale would make the operator
differ per sample and the inverse problem would stop being well defined.

This script projects a sample of slices, takes a high percentile of the
sinogram values (a percentile, not the max, so one freak slice does not set
the scale for the whole dataset), and prints the number to paste into
`data.sino_scale`.

Note: this deliberately does NOT go through LoDoPaBSinogramDataset. That class
divides by `data.sino_scale`, so using it here would compute the scale from
data the scale had already been applied to. The projection is repeated instead.

The scale is computed on the TRAIN split only. Test data must not influence any
constant the model is fitted with, and the splits are patient-disjoint, so the
train scale is the honest one to use everywhere.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="which split folder to read (default train)")
    ap.add_argument("--n-slices", type=int, default=128,
                    help="how many slices to sample (default 128)")
    ap.add_argument("--percentile", type=float, default=99.9,
                    help="percentile of sinogram values to use (default 99.9)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]

    # data.root holds train/, val/ and test/ subfolders, one per LoDoPaB part.
    # The file names inside differ ("..._train_000.hdf5" vs "..._validation_000.hdf5"),
    # so match on extension and let the folder decide the split.
    root = Path(d["root"]) / args.split
    files = sorted(glob.glob(str(root / "*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"No .hdf5 files in {root}\n"
            f"data.root must hold train/, val/ and test/ subfolders."
        )
    print(f"found {len(files)} hdf5 file(s) in {root}")

    size = int(d.get("size", 181))
    n_angles = int(d.get("n_angles", 256))
    a0 = float(d.get("angle_start_deg", 0.0))
    a1 = float(d.get("angle_end_deg", 180.0))
    circle = bool(d.get("circle", True))

    from deepinv.physics import Tomography
    step = (a1 - a0) / n_angles
    angles = a0 + step * torch.arange(n_angles, dtype=torch.float32)
    op = Tomography(
        angles=angles, img_width=size, circle=circle,
        normalize=bool(d.get("normalize_operator", False)), fan_beam=False,
        adjoint_via_backprop=bool(d.get("adjoint_via_backprop", True)),
        device="cpu", dtype=torch.float32,
    )

    # Spread the sample across files rather than taking the first N from file 0,
    # so one patient does not dominate.
    per_file = max(1, args.n_slices // len(files))
    vals = []
    img_min, img_max = np.inf, -np.inf
    n_done = 0

    for path in files:
        if n_done >= args.n_slices:
            break
        with h5py.File(path, "r") as h5:
            n = h5["data"].shape[0]
            rows = np.linspace(0, n - 1, min(per_file, n)).astype(int)
            arr = h5["data"][rows]                       # (k, 362, 362)

        x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(1)

        # The last file of a LoDoPaB part is padded with all-zero slices. Drop
        # them: a blank slice contributes nothing but zeros to the percentile.
        keep = x.flatten(1).abs().max(dim=1).values > 0
        if not bool(keep.all()):
            x = x[keep]
        if x.shape[0] == 0:
            continue

        img_min = min(img_min, float(x.min()))
        img_max = max(img_max, float(x.max()))

        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear",
                              align_corners=False, antialias=True)
        with torch.no_grad():
            s = op.A(x)
        vals.append(s.flatten().numpy())
        n_done += x.shape[0]

    v = np.concatenate(vals)
    p = float(np.percentile(v, args.percentile))

    print()
    print("=" * 60)
    print(f"split                   : {args.split}")
    print(f"slices projected        : {n_done}")
    print(f"ground-truth image range: [{img_min:.4f}, {img_max:.4f}]")
    print(f"sinogram shape          : {tuple(s.shape[1:])}")
    print(f"sinogram min / max      : {v.min():.4f} / {v.max():.4f}")
    print(f"sinogram {args.percentile}th pct     : {p:.4f}")
    print("=" * 60)
    print()
    print("Paste this into your config under data:")
    print()
    print(f"  sino_scale: {p:.4f}")
    print()
    print("After scaling, sinograms land in roughly [-1, 1]:")
    scaled = v / p * 2.0 - 1.0
    print(f"  scaled range : [{scaled.min():.3f}, {scaled.max():.3f}]")
    frac = float((np.abs(scaled) <= 1.0).mean())
    print(f"  fraction inside [-1, 1] : {100*frac:.2f}%")


if __name__ == "__main__":
    main()
