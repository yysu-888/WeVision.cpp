#!/usr/bin/env python3
"""Compare a [3,L] C++ M-RoPE dump to Python's [3,1,L] trace tensor."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(f"Usage: {sys.argv[0]} REFERENCE_TRACE OUTPUT.i64")
root = Path(sys.argv[1]); manifest = json.loads((root / "manifest.json").read_text())
record = manifest["tensors"]["text.position_ids"]
expected = np.load(root / record["file"]).astype(np.int64)[:, 0, :]
actual = np.fromfile(sys.argv[2], dtype=np.int64)
if actual.size != expected.size:
    raise SystemExit(f"FAIL: expected {expected.size} values, got {actual.size}")
actual = actual.reshape(expected.shape)
if not np.array_equal(expected, actual):
    axis, index = np.argwhere(expected != actual)[0]
    raise SystemExit(f"FAIL: axis={axis}, position={index}, expected={expected[axis,index]}, actual={actual[axis,index]}")
print(f"PASS: M-RoPE positions [3,{expected.shape[1]}] exactly match")
