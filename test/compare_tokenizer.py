#!/usr/bin/env python3
"""Compare the fixed Stage-5 Qwen chat prompt against a Python trace."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(f"Usage: {sys.argv[0]} REFERENCE_TRACE OUTPUT.i32")
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
record = manifest["tensors"]["input.input_ids"]
expected = np.load(root / record["file"]).astype(np.int32).reshape(-1)
actual = np.fromfile(sys.argv[2], dtype=np.int32)
if not np.array_equal(expected, actual):
    mismatch = np.flatnonzero(expected != actual)
    index = int(mismatch[0]) if mismatch.size else min(expected.size, actual.size)
    raise SystemExit(f"FAIL: expected {expected.size} ids, got {actual.size}; first mismatch at {index}: expected={expected[index] if index < expected.size else None}, actual={actual[index] if index < actual.size else None}")
print(f"PASS: {actual.size} token ids exactly match input.input_ids")
