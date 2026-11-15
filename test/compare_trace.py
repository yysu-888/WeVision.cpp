#!/usr/bin/env python3
"""Compare Python (.npy) and C++ (raw .bin) WeVision trace manifests.

Both trace formats use ``manifest.json`` with a ``tensors`` object. Python
traces carry NumPy dtypes; C++ FileTraceSink currently writes ggml dtype names
and raw contiguous bytes. This tool normalizes both representations and stops
at the first mismatching tensor, which keeps graph-alignment debugging local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CPP_DTYPES = {
    "f32": np.dtype("<f4"),
    "f16": np.dtype("<f2"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
}


def read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tensors = payload.get("tensors")
    if payload.get("format_version") != 1 or not isinstance(tensors, dict):
        raise ValueError(f"invalid trace manifest: {path}")
    return payload


def load_tensor(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / str(record["file"])
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    dtype_name = str(record["dtype"])
    dtype = CPP_DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported raw C++ trace dtype {dtype_name!r}: {path}")
    shape = tuple(int(value) for value in record["shape"])
    array = np.fromfile(path, dtype=dtype)
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"payload size does not match shape for {path}")
    return array.reshape(shape)


def compare(left: np.ndarray, right: np.ndarray, atol: float, rtol: float) -> tuple[bool, str]:
    # ggml represents a trailing unit dimension implicitly, so its manifest
    # cannot retain Python's leading batch=1 after reversed-dimension export.
    # This is intentionally limited to that exact singleton-batch case.
    while left.ndim > right.ndim and left.shape[0] == 1:
        left = left[0]
    while right.ndim > left.ndim and right.shape[0] == 1:
        right = right[0]
    if left.shape != right.shape:
        return False, f"shape {list(left.shape)} != {list(right.shape)}"
    if np.issubdtype(left.dtype, np.integer) and np.issubdtype(right.dtype, np.integer):
        if np.array_equal(left, right):
            return True, "exact integer match"
        count = int(np.count_nonzero(left != right))
        return False, f"integer mismatch at {count}/{left.size} values"

    lhs = left.astype(np.float64, copy=False)
    rhs = right.astype(np.float64, copy=False)
    diff = np.abs(lhs - rhs)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean((lhs - rhs) ** 2))) if diff.size else 0.0
    norm = float(np.linalg.norm(lhs.ravel()) * np.linalg.norm(rhs.ravel()))
    cosine = float(np.dot(lhs.ravel(), rhs.ravel()) / norm) if norm else 1.0
    ok = bool(np.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True))
    return ok, (
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"rmse={rmse:.8g} cosine={cosine:.10g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--require-exact-keys", action="store_true")
    args = parser.parse_args()

    lhs = read_manifest(args.left)
    rhs = read_manifest(args.right)
    lhs_tensors: dict[str, dict[str, Any]] = lhs["tensors"]
    rhs_tensors: dict[str, dict[str, Any]] = rhs["tensors"]
    lhs_names, rhs_names = set(lhs_tensors), set(rhs_tensors)
    if args.require_exact_keys and lhs_names != rhs_names:
        missing_left = sorted(rhs_names - lhs_names)
        missing_right = sorted(lhs_names - rhs_names)
        raise SystemExit(f"trace key mismatch: only-right={missing_left}, only-left={missing_right}")
    names = sorted(lhs_names & rhs_names)
    if not names:
        raise SystemExit("the two traces have no tensor names in common")

    for name in names:
        left = load_tensor(args.left, lhs_tensors[name])
        right = load_tensor(args.right, rhs_tensors[name])
        ok, summary = compare(left, right, args.atol, args.rtol)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {summary}")
        if not ok:
            raise SystemExit(1)
    print(f"OK: compared {len(names)} tensors")


if __name__ == "__main__":
    main()
