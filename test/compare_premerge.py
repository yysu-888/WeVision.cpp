#!/usr/bin/env python3
"""Compare a C++ merge-block token trace with Python's raster feature map."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("python_trace", type=Path)
p.add_argument("cpp_trace", type=Path)
p.add_argument("--layer", type=int, default=5)
p.add_argument("--atol", type=float, default=0.05)
a = p.parse_args()
py = json.loads((a.python_trace / "manifest.json").read_text())["tensors"][f"vision.block.{a.layer:02d}.premerge"]
expected = np.load(a.python_trace / py["file"])
cpp = json.loads((a.cpp_trace / "manifest.json").read_text())["tensors"][f"vision.block.{a.layer:02d}.premerge_tokens"]
tokens = np.fromfile(a.cpp_trace / cpp["file"], dtype=np.float32).reshape(cpp["shape"])
_, channels, height, width = expected.shape
merge = 2
actual = tokens.reshape(height // merge, width // merge, merge, merge, channels).transpose(0, 2, 1, 3, 4)
actual = actual.reshape(height, width, channels).transpose(2, 0, 1)[None]
diff = np.abs(actual - expected)
rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
cos = float(np.dot(actual.ravel(), expected.ravel()) / (np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel())))
print(f"shape={actual.shape} max_abs={diff.max():.7g} rmse={rmse:.7g} cosine={cos:.10f}")
raise SystemExit(0 if diff.max() <= a.atol else 1)
