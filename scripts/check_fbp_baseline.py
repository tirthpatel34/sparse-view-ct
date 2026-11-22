"""
How bad is plain FBP at the new geometry?

    python scripts/check_fbp_baseline.py --config configs/ct.yaml

Why this exists
---------------
The whole project rests on there being a gap between what a scanner gets from
few views and what the ground truth looks like. If filtered backprojection on
16 views already produces a decent image, there is nothing for the model to
win, and a 15-hour training run would tell us that very slowly.

So: take some TEST slices, project them, keep only every Nth angle, run FBP,
and measure the damage. A couple of minutes, and it tells you whether the
sampling ratio is aggressive enough before you commit to a real run.

Two reconstructions per ratio
-----------------------------
sparse  -- a Tomography operator built for ONLY the kept angles, reconstructing
           the kept columns. This is the real baseline: what a scanner that
           acquired N views would produce. The operator normalises for its own
           angle count, so the amplitude is correct.

zerofill -- the full 256-angle operator fed a sinogram whose dropped columns are
           zero. This is NOT a fair baseline: the filtering and normalisation
           assume every angle carries signal, so the result is scaled wrong and
           its PSNR is meaningless. It is printed only because it is the shape
           the model works in, so it is useful to see how different it looks.

Quote the sparse number. The scale constant data.sino_scale is deliberately not
applied: FBP wants physical line integrals, and the reconstruction then comes
back in ground-truth units, so PSNR means something.

CORRECTION, 16 Nov 2026: until then this script called op.A_dagger(y) with no
arguments. In deepinv 0.4.2 that is NOT FBP: it is a least-squares solve (CG on the
normal equations, up to 50 iterations, tol 1e-4, stopping only when EVERY slice in the
batch has converged), so the numbers depended on the batch size. The old table
(53.41 / 35.38 / 31.51 / 28.18 dB) came from that. Now:
  FBP      = op.A_dagger(y, fbp=True)                                   (the baseline)
  CGLS-50  = op.A_dagger(y, fbp=False, solver="CG", max_iter=50, tol=0.0)  (deterministic,
             batch-independent least squares; a second, stronger baseline)
A_dagger stores max_iter/tol on the operator, so every call passes them explicitly.
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


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def build_op(angles, size, d):
    from deepinv.physics import Tomography
    return Tomography(
        angles=angles, img_width=size, circle=bool(d.get("circle", True)),
        normalize=bool(d.get("normalize_operator", False)), fan_beam=False,
        adjoint_via_backprop=bool(d.get("adjoint_via_backprop", True)),
        device="cpu", dtype=torch.float32,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--ratios", type=int, nargs="+", default=[1, 8, 16, 32],
                    help="keep every Nth angle; 1 means the full scan")
    ap.add_argument("--out", default="fbp_baseline.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]

    root = Path(d["root"]) / args.split
    files = sorted(glob.glob(str(root / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {root}")

    size = int(d.get("size", 181))
    n_angles = int(d.get("n_angles", 256))
    a0 = float(d.get("angle_start_deg", 0.0))
    a1 = float(d.get("angle_end_deg", 180.0))

    step = (a1 - a0) / n_angles
    angles = a0 + step * torch.arange(n_angles, dtype=torch.float32)
    op_full = build_op(angles, size, d)

    # ---- load a spread of slices -------------------------------------------
    per_file = max(1, args.n_slices // len(files))
    chunks = []
    for path in files:
        with h5py.File(path, "r") as h5:
            n = h5["data"].shape[0]
            rows = np.linspace(0, n - 1, min(per_file, n)).astype(int)
            chunks.append(h5["data"][rows])
    x = torch.from_numpy(np.concatenate(chunks)).float().unsqueeze(1)
    x = x[x.flatten(1).abs().max(dim=1).values > 0][: args.n_slices]

    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
    print(f"{x.shape[0]} slices at {size} x {size}, {n_angles} angles\n")

    with torch.no_grad():
        sino = op_full.A(x)                 # (B, 1, n_det, n_angles)

    print(f"{'ratio':>5} {'views':>6} {'sparse FBP':>12} {'CGLS-50':>9} {'zerofill FBP':>13}")
    print("-" * 50)

    recons, scores = {}, {}
    for r in args.ratios:
        idx = torch.arange(0, n_angles, r)

        op_r = op_full if r == 1 else build_op(angles[idx], size, d)
        with torch.no_grad():
            rec = op_r.A_dagger(sino[..., idx], fbp=True)
            rec_ls = op_r.A_dagger(sino[..., idx], fbp=False, solver="CG",
                                   max_iter=50, tol=0.0)
            y = torch.zeros_like(sino)
            y[..., idx] = sino[..., idx]
            rec_zf = op_full.A_dagger(y, fbp=True)

        recons[r] = rec
        scores[r] = [psnr(rec[i], x[i]) for i in range(x.shape[0])]
        zf = np.mean([psnr(rec_zf[i], x[i]) for i in range(x.shape[0])])
        ls = np.mean([psnr(rec_ls[i], x[i]) for i in range(x.shape[0])])
        print(f"{r:>5} {len(idx):>6} {np.mean(scores[r]):>9.2f} dB "
              f"{ls:>6.2f} dB {zf:>10.2f} dB")

    print()
    for r in args.ratios:
        print(f"  ratio {r:>2}: min {np.min(scores[r]):.2f}, "
              f"max {np.max(scores[r]):.2f} dB")

    # ---- picture -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(4, x.shape[0])
    cols = 1 + len(args.ratios)
    fig, axes = plt.subplots(n_show, cols, figsize=(2.4 * cols, 2.6 * n_show))
    axes = np.atleast_2d(axes)
    for i in range(n_show):
        axes[i, 0].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title("ground truth", fontsize=9)
        for j, r in enumerate(args.ratios):
            axes[i, j + 1].imshow(recons[r][i, 0], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i, j + 1].set_title(f"{n_angles // r} views", fontsize=9)
            axes[i, j + 1].set_xlabel(f"{scores[r][i]:.1f} dB", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
