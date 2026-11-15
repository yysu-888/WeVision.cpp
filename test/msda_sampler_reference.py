#!/usr/bin/env python3
"""Write a deterministic MSDeformAttn sampler fixture and Python reference."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
# ggml layouts: value[D,H,T], locations[2,P*L,H,Q], weights[P*L,H,Q].
value = rng.normal(size=(2, 2, 10)).astype(np.float32)
locations = rng.uniform(-0.2, 1.2, size=(2, 4, 2, 3)).astype(np.float32)
weights = rng.normal(size=(4, 2, 3)).astype(np.float32)
# NumPy is last-dimension contiguous whereas ggml is first-dimension
# contiguous. Serialize logical tensors in the ggml [D,H,T], [2,PL,H,Q],
# [PL,H,Q] physical orders expected by the C++ fixture.
value.transpose(2, 1, 0).copy().tofile(root / "value.f32")
locations.transpose(3, 2, 1, 0).copy().tofile(root / "locations.f32")
weights.transpose(2, 1, 0).copy().tofile(root / "weights.f32")

out = np.zeros((4, 3), dtype=np.float32)
offsets = (0, 6)
shapes = ((2, 3), (2, 2))
for query in range(3):
    for head in range(2):
        for level, (height, width) in enumerate(shapes):
            feature = torch.from_numpy(value[:, head, offsets[level]:offsets[level] + height * width].reshape(2, height, width))[None]
            for point in range(2):
                index = level * 2 + point
                grid = torch.from_numpy(locations[:, index, head, query].reshape(1, 1, 1, 2) * 2 - 1)
                sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0, 0]
                out[head * 2:(head + 1) * 2, query] += weights[index, head, query] * sampled.numpy()
np.save(root / "expected.npy", out.T) # trace exports Q,H
