#!/usr/bin/env python3
"""Compare C++ raw FP32 vision patches with a Python reference trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="trace directory containing manifest.json")
    parser.add_argument("actual", type=Path, help="C++ float32 tensor dumped by wevision-vision-dump")
    # Pillow and the C++ floating-point antialiased cubic implementations have
    # small boundary/rounding differences. RMSE is the meaningful aggregate
    # check; a single RGB value can differ by up to 8/255 after normalization.
    parser.add_argument("--atol", type=float, default=0.07)
    parser.add_argument("--rmse", type=float, default=0.01)
    args = parser.parse_args()

    manifest = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    item = manifest["tensors"]["input.pixel_values"]
    expected = np.load(args.reference / item["file"])
    actual = np.fromfile(args.actual, dtype=np.float32)
    if actual.size != expected.size:
        print(f"FAIL shape: expected {expected.shape} ({expected.size}), actual {actual.size}")
        return 1
    actual = actual.reshape(expected.shape)
    diff = np.abs(actual - expected)
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    max_abs = float(diff.max())
    cosine = float(np.dot(actual.ravel(), expected.ravel()) /
                   (np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel())))
    print(f"shape={actual.shape} max_abs={max_abs:.7g} rmse={rmse:.7g} cosine={cosine:.10f}")
    if max_abs > args.atol or rmse > args.rmse:
        print(f"FAIL tolerances: max_abs<={args.atol}, rmse<={args.rmse}")
        return 1
    print("PASS (difference is bounded to image-resize implementation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
