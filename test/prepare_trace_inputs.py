#!/usr/bin/env python3
"""Extract C++ smoke-test inputs from a Python reference trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(root: Path, record: dict[str, object]) -> np.ndarray:
    path = root / str(record["file"])
    if path.suffix != ".npy":
        raise ValueError(f"Python reference tensor must be .npy: {path}")
    return np.load(path, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.trace / "manifest.json").read_text(encoding="utf-8"))
    tensors: dict[str, dict[str, object]] = manifest["tensors"]
    ids = load(args.trace, tensors["input.input_ids"]).astype("<i4", copy=False)
    vision = load(args.trace, tensors["vision.merger.output"]).astype("<f4", copy=False)
    args.output.mkdir(parents=True, exist_ok=True)
    ids.tofile(args.output / "input_ids.i32")
    vision.tofile(args.output / "vision.merger.output.f32")
    for index in range(3):
        name = f"vision.deepstack.{index}.output"
        if name in tensors:
            load(args.trace, tensors[name]).astype("<f4", copy=False).tofile(
                args.output / f"vision.deepstack.{index}.output.f32"
            )
    for name, filename in (("mask.obj_hidden", "mask.obj_hidden.f32"), ("mask.boxes01", "mask.boxes01.f32")):
        if name in tensors:
            load(args.trace, tensors[name]).astype("<f4", copy=False).tofile(args.output / filename)
    if "mask.cnn_rgb" in tensors:
        load(args.trace, tensors["mask.cnn_rgb"]).astype("<f4", copy=False).tofile(args.output / "mask.cnn_rgb.f32")
    for name, filename in (("mask.transformer.output", "mask.transformer.output.f32"),
                           ("mask.pixel.fpn", "mask.pixel.fpn.f32"),
                           ("mask.pixel.feature", "mask.pixel.feature.f32"),
                           ("mask.coarse_logits", "mask.coarse_logits.f32")):
        if name in tensors:
            load(args.trace, tensors[name]).astype("<f4", copy=False).tofile(args.output / filename)
    for layer in (5, 11, 17, 23):
        name = f"vision.block.{layer:02d}.premerge"
        if name in tensors:
            load(args.trace, tensors[name]).astype("<f4", copy=False).tofile(args.output / f"{name}.f32")


if __name__ == "__main__":
    main()
