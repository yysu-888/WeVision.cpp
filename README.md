# WeVision.cpp

Standalone `ggml`/C++17 inference engine for **WeVisOne-VL** ([2B](https://huggingface.co/tencent/WeVisOne-VL-2B) and [4B](https://huggingface.co/tencent/WeVisOne-VL-4B)) with a pure C public API.

- **Tasks**: `detect`, `point`, `count`, `ocr`, `segment`, `matting`
- **Backends**: CPU, Metal (Apple Silicon), CUDA (NVIDIA), selected via `--backend cpu|metal|cuda`
- **Interfaces**: CLI (`wevision-cli`), C API (`include/wevision.h`), HTTP server (`wevision-serve`)

## Visual Demos

All examples below were generated with `WeVisOne-VL-2B` (`q4_k`, Metal backend) using `wevision-cli`.

### 1. Detection (`--task detect`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/dogs.jpg \
  --backend metal --task detect --query "dog"  --min-pixels $((128 * 128)) \
  --max-pixels $((512 * 512)) --visualize asset/dogs-detect.png 
```

| Input ([`images/dogs.jpg`](images/dogs.jpg)) | Output (`4` boxes: `dog`) |
| :---: | :---: |
| ![Dogs Input](images/dogs.jpg) | ![Dogs Detect](asset/dogs-detect.png) |

### 2. Pointing (`--task point`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/chess.jpg \
  --backend metal --task point --query "pawn" \
  --min-pixels 200704 --max-pixels 200704 --visualize asset/chess-point.png
```

| Input ([`images/chess.jpg`](images/chess.jpg)) | Output (`30` points: `pawn`) |
| :---: | :---: |
| ![Chess Input](images/chess.jpg) | ![Chess Point](asset/chess-point.png) |

### 3. Counting (`--task count`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/boys.jpg  --min-pixels $((128 * 128)) \
  --max-pixels $((512 * 512)) --backend metal --task count --query "boy" --visualize asset/boys-count.png
```

| Input ([`images/boys.jpg`](images/boys.jpg)) | Output (`6` instances) |
| :---: | :---: |
| ![Boys Input](images/boys.jpg) | ![Boys Count](asset/boys-count.png) |

### 4. OCR (`--task ocr`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/coffee_sign.jpg \
  --backend metal --task ocr --min-pixels $((128 * 128)) --max-pixels $((512 * 512)) --visualize asset/coffee-ocr.png
```

| Input ([`images/coffee_sign.jpg`](images/coffee_sign.jpg)) | Output (`13` text boxes) |
| :---: | :---: |
| ![Coffee Sign Input](images/coffee_sign.jpg) | ![Coffee OCR](asset/coffee-ocr.png) |

Recognized text: `REAL`, `COFFEE`, `&`, `Tea`, `Cafe`, `10`, `METERS`, `AHEAD`, `TURN`, `RIGHT`, `IN`, `SMALL`, `ALLEY`.

### 5. Segmentation (`--task segment`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/boys.jpg \
  --backend metal --task segment --query "boy" --min-pixels $((128 * 128)) --max-pixels $((512 * 512))  \
  --visualize asset/boys-segment.png --output-prefix asset/boys-seg
```

| Input ([`images/boys.jpg`](images/boys.jpg)) | Instance Overlay (`6` masks) | Combined Binary Mask (`6` masks merged) |
| :---: | :---: | :---: |
| ![Boys Input](images/boys.jpg) | ![Boys Segment](asset/boys-segment.png) | ![Boys Combined Masks](asset/boys-seg-masks.png) |

### 6. Matting (`--task matting`)

```sh
./build-metal/wevision-cli models/WeVisOne-VL-2B_q4_k.gguf images/seabreeze.jpg \
  --backend metal --task matting --query "boy" --min-pixels $((128 * 128)) --max-pixels $((512 * 512)) \
  --visualize asset/seabreeze-overlay.png --output-prefix asset/seabreeze
```

| Input ([`images/seabreeze.jpg`](images/seabreeze.jpg)) | Detection & Mask Overlay | Matting Alpha | Binary Mask |
| :---: | :---: | :---: | :---: |
| ![Input](images/seabreeze.jpg) | ![Overlay](asset/seabreeze-overlay.png) | ![Alpha](asset/seabreeze-alpha-0.png) | ![Mask](asset/seabreeze-mask-0.png) |

## Prerequisites

- CMake >= 3.16 and a C++17 compiler (`clang++` / `g++`)
- `third_party/ggml` is vendored in the repository (no extra submodule checkout required)
- Optional: Apple Xcode Command Line Tools (Metal), NVIDIA CUDA Toolkit with `nvcc` on `PATH` (CUDA), Python 3 + `safetensors` (for GGUF conversion)

## Build

```sh
cmake -S . -B build-cpu -DWEVISION_GGML_METAL=OFF                          # CPU
cmake -S . -B build-metal                                                  # Metal (Apple, default)
cmake -S . -B build-cuda -DWEVISION_GGML_CUDA=ON -DWEVISION_GGML_METAL=OFF # CUDA
cmake --build build-metal -j                                               # build whichever tree you configured
ctest --test-dir build-metal --output-on-failure
```

CUDA needs the toolkit with `nvcc` on `PATH`; a fresh tree pays a one-time JIT warm-up on first inference. CUDA binaries also accept `--backend cpu` for numerical A/B testing.

## Convert to GGUF

Download either [WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) or [WeVisOne-VL-4B](https://huggingface.co/tencent/WeVisOne-VL-4B) from Hugging Face and convert to GGUF:

```sh
huggingface-cli download tencent/WeVisOne-VL-2B --local-dir model_2b
# or: huggingface-cli download tencent/WeVisOne-VL-4B --local-dir model_4b

python3 -m pip install safetensors                          # one-time
python3 convert_wevisone_to_gguf.py \
    --checkpoint model_2b --output model_gguf/wevision-vl-2b-bf16.gguf
```

Main weights are written as BF16 (`--main-dtype f32` for an F32 baseline; the loader accepts both); the `wv.mask.*` decoder is always F32. A sibling `.manifest.json` records every tensor's dtype, ggml type, and shape.

## Quantize

```sh
./build-metal/wevision-quantize IN.gguf OUT.gguf q8_0      # or q4_k
./build-metal/wevision-quantize OUT.gguf --verify
```

Quantizes 2D linear weights under `wv.text.*` / `wv.vision.*`; keeps `wv.mask.*`, norms, biases, and embeddings unquantized (and the logits head unquantized for `q8_0`). Measured on 2B: `q8_0` halves VRAM (`4.9 -> 2.5 GB`), gives ~3.3x faster CPU decode, and stays within 0.5% of full precision. `q4_k` is experimental (possible detection drift or degraded masks) — prefer `q8_0` when accuracy is critical.

## CLI

```sh
./build-metal/wevision-cli model_gguf/wevision-vl-2b-q8_0.gguf images/seabreeze.jpg \
    --task matting --query "boy" --backend metal \
    --min-pixels 16384 --max-pixels 262144 --max-tokens 64 \
    --output-prefix /tmp/boy --visualize /tmp/boy-overlay.png
```

Reads JPEG/PNG/BMP/PPM; prints text, boxes, points, count, and stage timings; `segment`/`matting` tasks also write `<prefix>-mask-<i>.png` and `<prefix>-alpha-<i>.png`. Common flags:

- `--task detect|point|count|ocr|segment|matting`
- `--min-pixels N` / `--max-pixels N`: dynamic ViT pixel budget
- `--max-tokens N`: generation cap
- `--repeat N`: run `N` iterations (`run=1+` reflects steady-state latency)
- `--visualize PATH`: render boxes/masks overlay PNG

Run `./build-metal/wevision-cli` without arguments for all options.

## C API

The public C interface is declared in [`include/wevision.h`](include/wevision.h):

```c
#include "wevision.h"

wevision_model *model = NULL;
wevision_model_load("model_gguf/wevision-vl-2b-q8_0.gguf", &model);

wevision_runtime_options opts = {WEVISION_BACKEND_METAL, 0, 128};
wevision_runtime *rt = NULL;
wevision_runtime_create(model, &opts, &rt);
wevision_model_free(model); // safe to free after runtime creation

wevision_request req = {
    .image = {rgb_bytes, width, height, 0},
    .prompt = "boy",
    .task = WEVISION_TASK_MATTING,
    .vision_min_pixels = 128 * 128,
    .vision_max_pixels = 512 * 512,
};
wevision_result res = {0};
if (wevision_predict(rt, &req, &res) == WEVISION_OK) {
    // inspect res.text, res.boxes[i].{x1,y1,x2,y2,mask,alpha}
    wevision_result_free(&res);
}
wevision_runtime_free(rt);
```

## HTTP service

```sh
./build-metal/wevision-serve --model model_gguf/wevision-vl-2b-q8_0.gguf \
    --backend metal --host 0.0.0.0 --port 8717
./build-metal/wevision-client --url http://127.0.0.1:8717 \
    --image images/seabreeze.jpg --task matting --query boy \
    --save-prefix /tmp/mat
curl -s -X POST http://127.0.0.1:8717/infer -H 'Content-Type: application/json' \
    -d '{"image_path":"images/seabreeze.jpg","task":"detect","query":"boy"}'
```

`POST /infer` accepts `image_path`, `task` (`detect|point|count|ocr|segment|matting`), and optional `query`/`prompt`, `max_tokens`, `min_pixels`, `max_pixels`, `mask_threshold`. The JSON response returns `text`, `boxes`, `points`, `count`, stage timings, and base64-encoded PNG `mask`/`alpha` planes. The model loads once at startup and inference is serialized across requests.

## Performance

### CUDA (`WeVisOne-VL-2B`, `q4_k`, `seabreeze.jpg`)

Inference only; single B200 with concurrent tenants, so numbers are conservative:

| task | WeVisOne-VL-2B |
|---|---|
| detect | 1037 ms |
| point | 1060 ms |
| count | 1066 ms |
| ocr | 11678 ms (hit 256-token cap) |
| segment | 2593 ms |
| matting | 2135 ms |

Non-generative stages hold at 1–1.5 s; the mask stage runs 500–800 ms; decode scales linearly with generated tokens.

### Apple Silicon Metal (`WeVisOne-VL-2B`, `q4_k`, `seabreeze.jpg`, Apple M2)

Steady-state (`--min-pixels 16384 --max-pixels 262144`):

| Stage | Latency |
|---|---|
| `preprocess` | ~12 ms |
| `vision` | ~575–750 ms |
| `prefill` | ~560–700 ms |
| `decode` (11 tokens) | ~260–430 ms (`25–42 tok/s`) |
| `mask` (`s4 + s2 + s1`) | ~2950–3370 ms |
| `postprocess` | ~3 ms |

## Acknowledgements

- [ggml](https://github.com/ggml-org/ggml) for the tensor and backend runtime
- [WeVisOne-VL](https://huggingface.co/tencent/WeVisOne-VL-2B) ([4B](https://huggingface.co/tencent/WeVisOne-VL-4B)) model family
