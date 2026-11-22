"""
LoDoPaB-CT dataset -- returns SINOGRAMS, not images.

Why sinograms: the inverse problem here is posed in measurement space, "fill in
the missing angle columns", so each sample is the full 256-angle sinogram and
the sparse measurement is made from it by masking angle columns.

    .hdf5 slice (362x362)
        -> downsample to 181x181
        -> Radon transform, 256 angles over [0, 180)
        -> scale to [-1, 1]
        -> (1, 256, 256)  = (channel, detector cell, angle)

Splits: LoDoPaB ships train / validation / test as separate parts, and they are
patient-disjoint by construction (patient IDs 0-631, 632-691, 692-751). Each
split therefore reads its OWN folder under data.root:

    root/train/ground_truth_train_*.hdf5
    root/val/ground_truth_validation_*.hdf5
    root/test/ground_truth_test_*.hdf5

A subset of files per folder is fine -- any subset of the train part is still
patient-disjoint from val and test. The earlier version cut one part per slice,
which put the same patient on both sides of the split.

Scaling note: the sinogram is divided by ONE fixed constant (`data.sino_scale`),
the same for every sample and every split. It has to be fixed -- a per-image
scale would make the forward operator different for each sample, which breaks
the inverse problem. Run scripts/compute_sino_scale.py to find the number.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from deepinv.physics import Tomography


class LoDoPaBSinogramDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        super().__init__()
        d = cfg["data"]
        self.dcfg = d
        self.split = split

        # ---- which .hdf5 files to read -------------------------------------
        # The folder IS the split. No shuffling, no fractions, no leakage.
        subdir = {"train": "train", "val": "val",
                  "validation": "val", "test": "test"}
        if split not in subdir:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        root = Path(d["root"]) / subdir[split]
        files = sorted(glob.glob(str(root / "*.hdf5")))
        if not files:
            raise FileNotFoundError(
                f"No .hdf5 files in {root}\n"
                f"data.root must hold train/, val/ and test/ subfolders."
            )
        self.files: List[str] = files

        # ---- build a flat index of (file position, row within that file) ----
        # We do NOT load the slices here: 2560 x 362 x 362 floats is ~1.3 GB.
        #
        # LoDoPaB stores 128 slices per file. The LAST file of each part is
        # padded with all-zero "slices" to reach 128 (validation has 3522 real
        # slices but 28 x 128 = 3584 rows). A blank slice gives a blank
        # sinogram, which carries no information and inflates metrics, so we
        # skip it. Only the final file of a part can be padded; we check the
        # last file we actually hold, which is cheap and catches it when the
        # subset happens to include the end of a part.
        index: List[Tuple[int, int]] = []
        n_dropped = 0
        for fi, path in enumerate(files):
            with h5py.File(path, "r") as h5:
                n = h5["data"].shape[0]
                if fi == len(files) - 1:
                    # Max |value| per row, one row at a time (keeps memory low).
                    keep = [row for row in range(n)
                            if float(np.abs(h5["data"][row]).max()) > 0.0]
                else:
                    keep = list(range(n))
            n_dropped += n - len(keep)
            index.extend((fi, row) for row in keep)

        print(f"[lodopab] {split}: {len(index)} slices in {len(files)} files "
              f"({n_dropped} empty padding rows skipped)")

        self.items = index

        max_len = d.get("max_len", None)
        if max_len is not None:
            self.items = self.items[: int(max_len)]

        # ---- geometry ------------------------------------------------------
        self.size = int(d.get("size", 181))
        self.n_angles = int(d.get("n_angles", 256))
        self.angle_start = float(d.get("angle_start_deg", 0.0))
        self.angle_end = float(d.get("angle_end_deg", 180.0))
        self.circle = bool(d.get("circle", True))
        self.adjoint_via_backprop = bool(d.get("adjoint_via_backprop", True))
        self.normalize_operator = bool(d.get("normalize_operator", False))

        scale = d.get("sino_scale", None)
        if scale is None:
            raise ValueError(
                "data.sino_scale is not set.\n"
                "The sinogram has to be rescaled into [-1, 1] by a FIXED constant.\n"
                "Run:  python scripts/compute_sino_scale.py --config <your config>\n"
                "then paste the number it prints into data.sino_scale."
            )
        self.sino_scale = float(scale)

        # Opened lazily, per DataLoader worker. h5py handles and CUDA-free torch
        # ops do not survive a fork, so they must not be created in __init__.
        self._h5: Dict[int, Any] = {}
        self._op = None

    # ------------------------------------------------------------------ util
    def _handle(self, file_idx: int):
        h = self._h5.get(file_idx)
        if h is None:
            h = h5py.File(self.files[file_idx], "r")
            self._h5[file_idx] = h
        return h

    def _operator(self):
        if self._op is None:
            # Tomography imported at module level (see top) so deepinv is in
            # sys.modules before the DataLoader forks - a forked worker cannot
            # re-initialize CUDA, and importing deepinv triggers a GPU probe.
            step = (self.angle_end - self.angle_start) / self.n_angles
            angles = self.angle_start + step * torch.arange(
                self.n_angles, dtype=torch.float32
            )
            self._op = Tomography(
                angles=angles,
                img_width=self.size,
                circle=self.circle,
                normalize=self.normalize_operator,
                fan_beam=False,
                adjoint_via_backprop=self.adjoint_via_backprop,
                device="cpu",
                dtype=torch.float32,
            )
        return self._op

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.items)

    def load_image(self, idx: int) -> torch.Tensor:
        """The ground-truth IMAGE for item `idx`, (1, size, size), in LoDoPaB's own
        [0, 1] units. This is exactly the image __getitem__ projects; use it to
        score reconstructions in image space."""
        file_idx, row = self.items[idx]
        arr = self._handle(file_idx)["data"][row]            # (362, 362) float32

        x = torch.from_numpy(np.ascontiguousarray(arr)).float()
        x = x.unsqueeze(0).unsqueeze(0)                      # (1, 1, 362, 362)

        # antialias=True low-pass filters before resampling. Without it you alias
        # fine detail into the small image and quietly corrupt everything after.
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                              align_corners=False, antialias=True)
        return x[0]                                          # (1, size, size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.load_image(idx).unsqueeze(0)                # (1, 1, size, size)

        with torch.no_grad():
            sino = self._operator().A(x)                     # (1, 1, n_det, n_angles)

        sino = sino / self.sino_scale                        # ~[0, 1]
        sino = sino * 2.0 - 1.0                              # -> [-1, 1]
        return sino[0]                                       # (1, n_det, n_angles)
