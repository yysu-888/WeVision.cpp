#!/usr/bin/env python3
"""Convert a released WeVisOne-VL checkpoint into a self-contained GGUF file.

The converter keeps the released main model in BF16 by default and the
WeVisOne pixel decoder in F32. ``--main-dtype f32`` creates an alignment
baseline that avoids lower-precision matmul drift; quantization belongs in a
later, separately validated pass.

The output uses only ``wv.*`` tensor names. C++ model code must bind these
logical names and never depend on Hugging Face checkpoint naming.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

import torch
from safetensors import safe_open


GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGUF scalar metadata types.
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_STRING = 8
GGUF_ARRAY = 9

# ggml_type values from third_party/ggml/include/ggml.h.
GGML_TYPE_F32 = 0
GGML_TYPE_BF16 = 30


@dataclass(frozen=True)
class TensorPlan:
    source: str
    target: str
    shape: tuple[int, ...]
    ggml_shape: tuple[int, ...]
    ggml_type: int
    dtype: str
    byte_count: int
    offset: int
    layout: str = "reverse_shape_for_ggml"


def align(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_exact(handle: BinaryIO, value: bytes) -> None:
    written = handle.write(value)
    if written != len(value):
        raise OSError(f"short write: expected {len(value)} bytes, wrote {written}")


def write_u32(handle: BinaryIO, value: int) -> None:
    write_exact(handle, struct.pack("<I", value))


def write_f32(handle: BinaryIO, value: float) -> None:
    write_exact(handle, struct.pack("<f", value))


def write_i32(handle: BinaryIO, value: int) -> None:
    write_exact(handle, struct.pack("<i", value))


def write_u64(handle: BinaryIO, value: int) -> None:
    write_exact(handle, struct.pack("<Q", value))


def write_i64(handle: BinaryIO, value: int) -> None:
    write_exact(handle, struct.pack("<q", value))


def write_string(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    write_u64(handle, len(encoded))
    write_exact(handle, encoded)


def write_metadata_value(handle: BinaryIO, value: Any) -> None:
    if isinstance(value, str):
        write_i32(handle, GGUF_STRING)
        write_string(handle, value)
    elif isinstance(value, bool):
        write_i32(handle, 7)
        write_exact(handle, bytes((int(value),)))
    elif isinstance(value, int):
        write_i32(handle, GGUF_UINT32)
        write_u32(handle, value)
    elif isinstance(value, float):
        write_i32(handle, GGUF_FLOAT32)
        write_f32(handle, value)
    elif isinstance(value, list):
        if not value:
            raise ValueError("GGUF does not infer an element type for an empty array")
        if all(isinstance(item, int) for item in value):
            write_i32(handle, GGUF_ARRAY)
            write_i32(handle, GGUF_INT32)
            write_u64(handle, len(value))
            for item in value:
                write_i32(handle, item)
        elif all(isinstance(item, str) for item in value):
            write_i32(handle, GGUF_ARRAY)
            write_i32(handle, GGUF_STRING)
            write_u64(handle, len(value))
            for item in value:
                write_string(handle, item)
        else:
            raise TypeError(f"unsupported GGUF array value: {value[:2]!r}")
    else:
        raise TypeError(f"unsupported GGUF metadata value: {type(value)!r}")


def write_padding(handle: BinaryIO, alignment: int = GGUF_ALIGNMENT) -> None:
    padding = (-handle.tell()) % alignment
    if padding:
        write_exact(handle, b"\0" * padding)


def map_main_tensor(source: str) -> str:
    if source == "model.language_model.embed_tokens.weight":
        return "wv.text.token_embd.weight"
    if source == "model.language_model.norm.weight":
        return "wv.text.output_norm.weight"
    if source == "lm_head.weight":
        return "wv.text.output.weight"

    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", source)
    if match:
        index, suffix = match.groups()
        replacements = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        try:
            return f"wv.text.blk.{index}.{replacements[suffix]}"
        except KeyError as error:
            raise KeyError(f"unmapped text tensor: {source}") from error

    if source == "model.visual.pos_embed.weight":
        return "wv.vision.pos_embd.weight"
    if source.startswith("model.visual.patch_embed.proj."):
        return "wv.vision.patch_embed." + source.rsplit(".", 1)[-1]

    match = re.fullmatch(r"model\.visual\.blocks\.(\d+)\.(.+)", source)
    if match:
        index, suffix = match.groups()
        replacements = {
            "norm1.weight": "attn_norm.weight",
            "norm1.bias": "attn_norm.bias",
            "attn.qkv.weight": "attn_qkv.weight",
            "attn.qkv.bias": "attn_qkv.bias",
            "attn.proj.weight": "attn_output.weight",
            "attn.proj.bias": "attn_output.bias",
            "norm2.weight": "ffn_norm.weight",
            "norm2.bias": "ffn_norm.bias",
            "mlp.linear_fc1.weight": "ffn_up.weight",
            "mlp.linear_fc1.bias": "ffn_up.bias",
            "mlp.linear_fc2.weight": "ffn_down.weight",
            "mlp.linear_fc2.bias": "ffn_down.bias",
        }
        try:
            return f"wv.vision.blk.{index}.{replacements[suffix]}"
        except KeyError as error:
            raise KeyError(f"unmapped vision tensor: {source}") from error

    match = re.fullmatch(r"model\.visual\.deepstack_merger_list\.(\d+)\.(.+)", source)
    if match:
        index, suffix = match.groups()
        return f"wv.vision.deepstack.{index}.{suffix}"
    if source.startswith("model.visual.merger."):
        return "wv.vision.merger." + source.removeprefix("model.visual.merger.")
    raise KeyError(f"unmapped main-model tensor: {source}")


def map_mask_tensor(source: str) -> str:
    target = source
    target = re.sub(r"^transformer\.layers\.(\d+)", r"decoder.blk.\1", target)
    target = target.replace("pixel_fusion.", "pixel.")
    target = target.replace("highres_refine.", "alpha.")
    return "wv.mask." + target


def byte_count(shape: Sequence[int], ggml_type: int) -> int:
    element_count = 1
    for size in shape:
        element_count *= int(size)
    if ggml_type == GGML_TYPE_F32:
        return element_count * 4
    if ggml_type == GGML_TYPE_BF16:
        return element_count * 2
    raise ValueError(f"unsupported ggml type: {ggml_type}")


def main_tensor_plans(model_path: Path, start_offset: int, main_type: int) -> list[TensorPlan]:
    offset = start_offset
    plans: list[TensorPlan] = []
    with safe_open(model_path, framework="pt", device="cpu") as reader:
        for source in sorted(reader.keys()):
            shape = tuple(int(value) for value in reader.get_slice(source).get_shape())
            target = map_main_tensor(source)
            # ggml tensors have at most four dimensions. Qwen3-VL stores the
            # patch embedding as [out, in, temporal, height, width]. Fold the
            # contiguous in/temporal axes into one input-channel axis while
            # retaining its original byte order: [width, height, in*temporal,
            # out]. The C++ vision graph consumes this as a 2-D convolution.
            if source == "model.visual.patch_embed.proj.weight":
                out_channels, in_channels, temporal, height, width = shape
                ggml_shape = (width, height, temporal * in_channels, out_channels)
                layout = "conv3d_patch_oiTHW_to_ggml_hwctout"
            else:
                ggml_shape = tuple(reversed(shape))
                layout = "reverse_shape_for_ggml"
            count = byte_count(shape, main_type)
            plans.append(
                TensorPlan(
                    source=source,
                    target=target,
                    shape=shape,
                    ggml_shape=ggml_shape,
                    ggml_type=main_type,
                    dtype="f32" if main_type == GGML_TYPE_F32 else "bf16",
                    byte_count=count,
                    offset=offset,
                    layout=layout,
                )
            )
            offset = align(offset + count)
    return plans


def mask_tensor_plans(payload: dict[str, Any], start_offset: int) -> list[TensorPlan]:
    offset = start_offset
    plans: list[TensorPlan] = []
    state_dict = payload["state_dict"]
    for source in sorted(state_dict):
        tensor = state_dict[source]
        if tensor.dtype != torch.float32:
            raise TypeError(f"expected F32 mask tensor, got {source}: {tensor.dtype}")
        shape = tuple(int(value) for value in tensor.shape)
        count = byte_count(shape, GGML_TYPE_F32)
        plans.append(
            TensorPlan(
                source=source,
                target=map_mask_tensor(source),
                shape=shape,
                ggml_shape=tuple(reversed(shape)),
                ggml_type=GGML_TYPE_F32,
                dtype="f32",
                byte_count=count,
                offset=offset,
            )
        )
        offset = align(offset + count)
    return plans


def metadata_for(checkpoint: Path, mask_payload: dict[str, Any]) -> dict[str, Any]:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    vision = config["vision_config"]
    mask = mask_payload["cfg"]
    return {
        "general.architecture": "wevision.qwen3vl",
        "general.name": checkpoint.name,
        "general.alignment": GGUF_ALIGNMENT,
        "wevision.format_version": 1,
        "wv.text.vocab_size": text["vocab_size"],
        "wv.text.hidden_size": text["hidden_size"],
        "wv.text.layer_count": text["num_hidden_layers"],
        "wv.text.attention_head_count": text["num_attention_heads"],
        "wv.text.attention_head_count_kv": text["num_key_value_heads"],
        "wv.text.head_dim": text["head_dim"],
        "wv.text.intermediate_size": text["intermediate_size"],
        "wv.text.rope_theta": int(text["rope_parameters"]["rope_theta"]),
        "wv.text.rms_norm_eps": float(text.get("rms_norm_eps", 1e-6)),
        "wv.text.mrope_sections": list(text["rope_parameters"]["mrope_section"]),
        "wv.vision.hidden_size": vision["hidden_size"],
        "wv.vision.layer_count": vision["depth"],
        "wv.vision.head_count": vision["num_heads"],
        "wv.vision.intermediate_size": vision["intermediate_size"],
        "wv.vision.rope_theta": int(vision.get("rope_theta", 10000)),
        "wv.vision.layer_norm_eps": float(vision.get("layer_norm_eps", 1e-6)),
        "wv.vision.patch_size": vision["patch_size"],
        "wv.vision.temporal_patch_size": vision["temporal_patch_size"],
        "wv.vision.spatial_merge_size": vision["spatial_merge_size"],
        "wv.vision.deepstack_layers": list(vision["deepstack_visual_indexes"]),
        "wv.mask.architecture_version": mask["architecture_version"],
        "wv.mask.seg_token_id": int(mask_payload["seg_token_id"]),
        "wv.mask.spatial_merge_size": int(mask_payload["spatial_merge_size"]),
        "wv.mask.vision_feature_layers": list(mask["vision_feature_layers"]),
        "wv.mask.hidden_size": int(mask["hidden_dim"]),
        "wv.mask.feature_size": int(mask["mask_feature_dim"]),
        "wv.mask.msda_levels": int(mask["msda_num_levels"]),
        "wv.mask.msda_heads": int(mask["transformer_heads"]),
        "wv.mask.msda_points": int(mask["msda_num_points"]),
        "wv.mask.transformer_layers": int(mask["transformer_layers"]),
        "wv.mask.encoder_layers": int(mask.get("encoder_layers", 3)),
        "wv.mask.norm_groups": int(mask.get("norm_groups", 32)),
        "wv.mask.norm_eps": float(mask.get("norm_eps", 1e-5)),
        "wv.mask.cnn_image_size": int(mask["cnn_image_size"]),
        "wv.hf.config_json": (checkpoint / "config.json").read_text(encoding="utf-8"),
        "wv.hf.processor_config_json": (checkpoint / "processor_config.json").read_text(encoding="utf-8"),
        "wv.hf.tokenizer_config_json": (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"),
        "wv.hf.chat_template": (checkpoint / "chat_template.jinja").read_text(encoding="utf-8"),
        "wv.hf.tokenizer_json": (checkpoint / "tokenizer.json").read_text(encoding="utf-8"),
    }


def write_header(handle: BinaryIO, metadata: dict[str, Any], plans: Sequence[TensorPlan]) -> None:
    write_exact(handle, b"GGUF")
    write_u32(handle, GGUF_VERSION)
    write_i64(handle, len(plans))
    write_i64(handle, len(metadata))
    for key, value in metadata.items():
        write_string(handle, key)
        write_metadata_value(handle, value)
    for plan in plans:
        write_string(handle, plan.target)
        write_u32(handle, len(plan.ggml_shape))
        for size in plan.ggml_shape:
            write_i64(handle, size)
        write_i32(handle, plan.ggml_type)
        write_u64(handle, plan.offset)
    write_padding(handle)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint8).numpy().tobytes()
    if tensor.dtype == torch.float32:
        return tensor.numpy().tobytes()
    raise TypeError(f"unsupported source tensor dtype: {tensor.dtype}")


def write_tensor(handle: BinaryIO, tensor: torch.Tensor, plan: TensorPlan) -> None:
    if tuple(tensor.shape) != plan.shape:
        raise ValueError(f"shape changed for {plan.source}: {tuple(tensor.shape)} != {plan.shape}")
    if plan.ggml_type == GGML_TYPE_F32:
        payload = tensor.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    elif plan.ggml_type == GGML_TYPE_BF16:
        payload = tensor_bytes(tensor)
    else:
        raise ValueError(f"unsupported planned ggml type: {plan.ggml_type}")
    if len(payload) != plan.byte_count:
        raise ValueError(f"byte count mismatch for {plan.source}")
    write_exact(handle, payload)
    write_padding(handle)


def convert(checkpoint: Path, output: Path, manifest_path: Path, dry_run: bool, main_type: int) -> None:
    model_path = checkpoint / "model.safetensors"
    mask_path = checkpoint / "mask_decoder.pt"
    for path in (model_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    mask_payload = torch.load(mask_path, map_location="cpu", weights_only=True)
    if not isinstance(mask_payload, dict) or "state_dict" not in mask_payload:
        raise ValueError("mask_decoder.pt does not contain a state_dict")
    metadata = metadata_for(checkpoint, mask_payload)

    # GGUF offsets are relative to the aligned tensor-data section. Plan once
    # with offset zero; header length does not affect individual tensor offsets.
    main = main_tensor_plans(model_path, 0, main_type)
    mask_start = align(main[-1].offset + main[-1].byte_count) if main else 0
    mask = mask_tensor_plans(mask_payload, mask_start)
    plans = main + mask

    names = [plan.target for plan in plans]
    if len(names) != len(set(names)):
        raise ValueError("multiple source tensors map to the same GGUF name")

    manifest = {
        "format": "wevision-conversion-manifest-v1",
        "checkpoint": str(checkpoint.resolve()),
        "output": str(output.resolve()),
        "tensor_count": len(plans),
        "main_tensor_count": len(main),
        "mask_tensor_count": len(mask),
        "metadata": metadata,
        "tensors": [asdict(plan) for plan in plans],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        print(f"Dry run: {len(main)} main tensors + {len(mask)} mask tensors mapped")
        print(f"Manifest: {manifest_path}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        write_header(handle, metadata, plans)
        data_offset = handle.tell()
        if data_offset % GGUF_ALIGNMENT:
            raise AssertionError("GGUF tensor data is not aligned")
        with safe_open(model_path, framework="pt", device="cpu") as reader:
            for index, plan in enumerate(main, start=1):
                write_tensor(handle, reader.get_tensor(plan.source), plan)
                if index % 50 == 0 or index == len(main):
                    print(f"Wrote main tensor {index}/{len(main)}", flush=True)
        for index, plan in enumerate(mask, start=1):
            write_tensor(handle, mask_payload["state_dict"][plan.source], plan)
            if index % 50 == 0 or index == len(mask):
                print(f"Wrote mask tensor {index}/{len(mask)}", flush=True)
    temporary.replace(output)
    print(f"Wrote GGUF: {output}")
    print(f"Wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--main-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    main_type = GGML_TYPE_F32 if args.main_dtype == "f32" else GGML_TYPE_BF16
    convert(args.checkpoint, args.output, manifest, args.dry_run, main_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
