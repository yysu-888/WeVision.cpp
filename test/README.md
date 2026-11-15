# 可复现验证

`src/` 只包含库实现，所有验证入口都集中于本目录。测试不修改模型或
reference；临时 payload 统一写入 `/private/tmp`。

## 构建与基础检查

```sh
cmake -S . -B build-cpu -DWEVISION_GGML_METAL=OFF -DWEVISION_BUILD_TESTS=ON
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure

./build-cpu/wevision-inspect models/wevision-vl-2b-f16.gguf
./build-cpu/wevision-inspect models/wevision-vl-2b-f16.gguf --realize=cpu
```

Metal 机器可将 `WEVISION_GGML_METAL` 设为 `ON`，并执行
`--realize=metal`。没有可用 Metal command queue 的运行环境只能验证编译。

## Trace 基础设施

构图代码通过 `GraphSession::checkpoint()` 发布语义检查点；`FileTraceSink` 的 selector
可使用精确名或前缀（如 `text.*`、`vision.deepstack.*`）。未绑定 sink 时不添加任何调试
节点。选中但不在最终输出路径上的 FP32 tensor 会只在调试图中由零值依赖保活，因此可在
一次 backend compute 后安全导出。

```sh
rm -rf /private/tmp/wevision-trace-smoke
./build-cpu/wevision-trace-smoke models/wevision-vl-2b-f16.gguf /private/tmp/wevision-trace-smoke
```

生成的 `manifest.json` 应记录 `smoke.mask.query_builder.content_proj.bias`。
`compare_trace.py` 比较 Python `.npy` trace 与 C++ 原始 trace。

### 数据布局算子统计

设置 `WEVISION_OP_PROFILE=1` 会逐个统计计算图中的 `PERMUTE`、`CONT`、
`REPEAT` 和 `CONCAT`。每行包含节点序号、输入/输出 shape、类型、字节数和耗时，
图结束时还会输出各类算子的次数、累计输出字节数和累计耗时：

```sh
WEVISION_OP_PROFILE=1 ./build-metal/wevision-cli \
  models/wevision-vl-2b-f16.gguf image.ppm --backend metal --task segmentation
```

该模式会为每个目标节点单独提交并同步 backend，只用于定位热点；它会放大整图耗时，
不应作为端到端性能数据。默认关闭时，计算图的提交方式不变。

## 图像预处理对齐

Python reference 已在 `../WeVisOne-VL/reference/segmentation_cpu_128/`。
将测试图转换为无色彩变换的 PPM RGB（例如本机 ImageMagick），然后：

```sh
magick ../WeVisOne-VL/demo/examples/images/seabreeze.jpg /private/tmp/wevision-seabreeze.ppm
./build-cpu/wevision-vision-dump /private/tmp/wevision-seabreeze.ppm \
  /private/tmp/wevision-pixel-values.f32 16384 16384
/Users/yangsu/miniforge3/envs/exp/bin/python compare_preprocess.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128 /private/tmp/wevision-pixel-values.f32
```

预期 `grid_thw=[1,6,8]`、shape `[48,1536]`。Pillow 和 C++ 浮点
antialiased cubic 的边界舍入不同；当前阈值为 max abs `0.07`、RMSE `0.01`。

## Vision patch embedding

```sh
rm -rf /private/tmp/wevision-patch-smoke
./build-cpu/wevision-vision-patch-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-patch-smoke
```

其 manifest 中 `vision.patch_embed.output` 的期望 shape 为 `[48,1024]`、
dtype 为 `f32`。该测试验证 patch embedding 图构建、输入上传、真实权重读取和
trace 导出。

对齐已导出的 Python boundary 时，将 `input.pixel_values` 转为 raw FP32 后作为
第三个参数传入，再用通用比较器检查同名 tensor：

```sh
/Users/yangsu/miniforge3/envs/exp/bin/python -c \
  "import numpy as n; n.load('../WeVisOne-VL/reference/segmentation_cpu_128_patch/002_input_pixel_values.npy').tofile('/private/tmp/wevision-patches.f32')"
rm -rf /private/tmp/wevision-patch-real
./build-cpu/wevision-vision-patch-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-patch-real /private/tmp/wevision-patches.f32
/Users/yangsu/miniforge3/envs/exp/bin/python compare_trace.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_patch /private/tmp/wevision-patch-real
```

当前 released FP16 GGUF 对 Python CPU FP32 reference 的验收结果为 max abs
`0.00832`、RMSE `0.000513`、cosine `0.99999957`；差异来自权重精度，而不是
patch layout 或图构建。

同一工具还会导出并比较 `vision.positioned_input`：它覆盖 Qwen 的 merge-block
position IDs、48×48 learned position table 的四路 gather、双线性权重和与 patch
embedding 的相加。

## 完整 Vision 与 DeepStack

`vision_stack_smoke` 在 layer 5、11、17 建立 post-shuffle-norm 的 deepstack merger，
在 layer 23 建立 final merger。下列命令以同一原始 patch 输入验收四个关键输出边界：

```sh
rm -rf /private/tmp/wevision-stack23-deep
./build-cpu/wevision-vision-stack-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-stack23-deep /private/tmp/wevision-patches.f32 23
/Users/yangsu/miniforge3/envs/exp/bin/python compare_trace.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_deepstack \
  /private/tmp/wevision-stack23-deep --atol 0.5 --rtol 0.05
```

`vision.deepstack.0/1/2.output` 和 `vision.merger.output` 均须通过。测试图仅用一个
add 链保持独立的 deepstack 输出可达；该链不属于生产模型计算。

## Tokenizer 与多模态 Prompt

固定的系统提示、带单图的用户提示和 teacher-forced assistant 回复，应产生 Python
reference 中完全相同的 `input.input_ids`；该样例会把 12 个 vision tokens 展开为 12 个
`<|image_pad|>`：

```sh
rm -f /private/tmp/wevision-tokenizer.i32
./build-cpu/wevision-tokenizer-dump models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-tokenizer.i32
/Users/yangsu/miniforge3/envs/exp/bin/python compare_tokenizer.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_deepstack \
  /private/tmp/wevision-tokenizer.i32
```

期望输出为 `PASS: 63 token ids exactly match input.input_ids`，并报告
`<OBJ>=152678`。该工具不调用 Vision 或 Text 图，专用于隔离 tokenizer/template 边界。

## Text 输入 Embedding

该边界使用 Python reference 的 final Vision merger 输出，以隔离 token lookup 和 image-pad
替换，不把 Vision 数值误差混入 Text 验收：

```sh
./build-cpu/wevision-text-embedding-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-tokenizer.i32 /private/tmp/wevision-python-vision.f32 \
  /private/tmp/wevision-text-embed
/Users/yangsu/miniforge3/envs/exp/bin/python compare_trace.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_textembed \
  /private/tmp/wevision-text-embed --atol 0.01 --rtol 0.001
```

`text.token_embeddings` 应完全一致，`text.input_embeddings` 允许浮点写回误差；当前最大绝对误差为 `5.96e-08`。

## Text decoder block 诊断与验收

reference manifest 本身是 NumPy 格式；先由下列小工具导出 smoke test 所需的
raw token IDs 和 final Vision embedding。`qkv-smoke` 保留所有中间检查点，只在
decoder 对齐失败时使用；`full-smoke` 则只验收 block 输出，代表正常的边界验收。

```sh
rm -rf /private/tmp/wevision-text-input /private/tmp/wevision-text-block0-full
/Users/yangsu/miniforge3/envs/exp/bin/python prepare_trace_inputs.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_textrope /private/tmp/wevision-text-input
./build-cpu/wevision-text-block0-full-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-text-input/input_ids.i32 \
  /private/tmp/wevision-text-input/vision.merger.output.f32 \
  /private/tmp/wevision-text-block0-full
/Users/yangsu/miniforge3/envs/exp/bin/python compare_trace.py \
  ../WeVisOne-VL/reference/segmentation_cpu_128_textrope \
  /private/tmp/wevision-text-block0-full --atol 0.06 --rtol 0.01
```

当前 `text.block.00.output` 的结果为 max abs `0.05052`、cosine
`0.9999969`。当它不通过时，再执行 `wevision-text-block0-qkv-smoke` 并以
相同方式比较，检查点会定位到 norm、Q/K/V、Q/K norm、M-RoPE 和 attention。

完整 prefill 还需要三个 DeepStack embedding；它们在第 0、1、2 个 decoder
block 之后以加法注入 image-pad positions。以下命令导出所有 28 个 decoder
边界并完成验收：

```sh
USE_TF=0 /Users/yangsu/miniforge3/envs/exp/bin/python \
  ../WeVisOne-VL/demo/examples/export_reference_trace.py \
  --model ../WeVisOne-VL-2B --image ../WeVisOne-VL/demo/examples/images/seabreeze.jpg \
  --task segmentation --query person \
  --response '<ref>person</ref><box><10><71><750><1000><OBJ></box>' \
  --output /private/tmp/wevision-text-reference --capture-all-text-blocks \
  --skip-mask-decoder --min-pixels 16384 --max-pixels 16384
/Users/yangsu/miniforge3/envs/exp/bin/python prepare_trace_inputs.py \
  /private/tmp/wevision-text-reference /private/tmp/wevision-text-input
./build-cpu/wevision-text-prefill-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-text-input/input_ids.i32 \
  /private/tmp/wevision-text-input/vision.merger.output.f32 \
  /private/tmp/wevision-text-input/vision.deepstack.0.output.f32 \
  /private/tmp/wevision-text-input/vision.deepstack.1.output.f32 \
  /private/tmp/wevision-text-input/vision.deepstack.2.output.f32 \
  /private/tmp/wevision-text-prefill
/Users/yangsu/miniforge3/envs/exp/bin/python compare_trace.py \
  /private/tmp/wevision-text-reference /private/tmp/wevision-text-prefill \
  --atol 0.01 --rtol 0.001
```

当前 CPU FP32-activation baseline 的 38 个 text checkpoints 均通过；最终
`text.last_hidden` 为 max abs `0.10254`、cosine `1.0`。`build_text_linear`
在计算时临时将 BF16 GGUF weights 提升至 FP32，避免 28 层 residual stream 中的
累积误差；GGUF 的磁盘存储格式不变。

同一 prefill smoke 还会导出每层 `text.kv.XX.key/value` 并回读写入
`TextKVCache`，以及验收 `text.next_logits`。当前 logits 的 max abs 为
`1.42e-4`、cosine `1.0`，Python/C++ greedy next token 均为 `152670`。

`wevision-text-decode-smoke` 以这份 prefill cache 运行一个 `<ref>` token 的
增量 decode。Python reference 的 `--append-token-id` 会在已经渲染完成的
input IDs 末尾追加 token（而不是修改 chat message，避免改变 chat-template
中 `<|im_end|>` 的位置）。当前 decode hidden max abs `0.00562`，next logits
max abs `2.04e-4`，双方 greedy token 均为 `8987`。

### MSDeformAttn sampler

阶段 8 的任意坐标双线性采样由标准 ggml 算子组成，不依赖 custom op 或 CPU
fallback。CPU 与 Metal 使用同一计算图，并保持 PyTorch `grid_sample`（bilinear、
zero padding、`align_corners=False`）语义。下面的确定性夹具分别验证两个后端；
其输入文件已经按 ggml 的第一维连续布局序列化。

```sh
rm -rf /private/tmp/wevision-msda-fixture \
  /private/tmp/wevision-msda-cpu /private/tmp/wevision-msda-metal
/Users/yangsu/miniforge3/envs/exp/bin/python test/msda_sampler_reference.py \
  /private/tmp/wevision-msda-fixture
./build-cpu/wevision-msda-sampler-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-msda-fixture/value.f32 \
  /private/tmp/wevision-msda-fixture/locations.f32 \
  /private/tmp/wevision-msda-fixture/weights.f32 \
  /private/tmp/wevision-msda-cpu cpu
./build-metal/wevision-msda-sampler-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-msda-fixture/value.f32 \
  /private/tmp/wevision-msda-fixture/locations.f32 \
  /private/tmp/wevision-msda-fixture/weights.f32 \
  /private/tmp/wevision-msda-metal metal
```

当前 fixture 的 CPU/PyTorch 与 Metal/PyTorch 最大绝对误差均为 `1.19e-7`，
CPU/Metal 输出完全一致。完整 Mask transformer 也已在两个后端通过 trace 对齐；
最终输出 CPU/Python 最大绝对误差为 `9.54e-7`，Metal/Python 为 `2.02e-4`。

### Pixel graph-op fixtures

Mask 的全分辨率路径使用内部 `C,W,H` layout。以下 fixture 验证 CPU
与 Metal 的 `PixelShuffle(2)`、`bilinear(..., align_corners=False)` 和
Pixel Fusion adaptive average pooling 节点与 PyTorch 的一致性：

```sh
rm -rf /private/tmp/wevision-pixelops-reference /private/tmp/wevision-pixelops-cpp
/Users/yangsu/miniforge3/envs/exp/bin/python test/pixel_ops_reference.py \
  /private/tmp/wevision-pixelops-reference
./build-cpu/wevision-pixel-ops-smoke models/wevision-vl-2b-f16.gguf \
  /private/tmp/wevision-pixelops-reference/input.f32 \
  /private/tmp/wevision-pixelops-cpp
/Users/yangsu/miniforge3/envs/exp/bin/python test/compare_trace.py \
  /private/tmp/wevision-pixelops-reference /private/tmp/wevision-pixelops-cpp
```

Metal 可在最后追加 `metal` 参数并使用 `build-metal` 二进制。PixelShuffle
在 CPU、Metal 上均为 exact；bilinear resize 对 PyTorch 的最大绝对误差分别为
`4.77e-7`、`1.19e-7`，CPU/Metal 最大绝对差为 `4.32e-7`；adaptive average
pooling 两个后端对 PyTorch 的最大绝对误差均为 `5.96e-8`。

### Mask visual encoder 输入边界

`wevision-mask-encoder-inputs-smoke` 复用四份 `vision.block.*.premerge`
输入，验证 projection 之后的 flattened `src` 和 DINO 2-D sine/level `pos`。
Python trace 通过 `visual_encoder.encoder` 的 pre-hook 导出这两个边界；当前
CPU FP32 的 max abs 分别为 `1.01e-5` 与 `1.19e-7`。随后再从该边界接入
SpatialConv encoder 层，避免把 conv、LayerNorm 和输入布局问题混在一起排查。
当前三层 encoder 的 `mask.visual_encoder.memory` 已对齐，max abs `0.00135`、
cosine `0.99999997`。

### Metal alpha 2x upsampler

Metal 上的 `ConvTranspose2d(k=4,s=2,p=1)` 被等价展开为一个 3x3 direct
convolution，输出四个空间 phase，再复用 `PixelShuffle(2)`。CPU 仍执行原始
ConvTranspose 节点。确定性夹具同时构建两条路径并比较最终输出：

```sh
./build-metal/wevision-mask-conv-transpose-unit \
  models/wevision-vl-2b-f16.gguf
```

Apple M2 上当前最大绝对误差为 `3.28e-7`、平均绝对误差为 `3.06e-8`。
完整分割相对优化前 Metal baseline 的 8-bit alpha 最大差为 `2`、平均差为
`0.056`；二值 mask 变化 284 / 1,749,600 像素，均位于阈值边界。

S/1 在 Metal 上全程保持 `W,H,C`，普通卷积使用标准 ggml
`im2col(F16) + mul_mat`，GroupNorm、residual 和输出仍为 F32。相对原始
CWH/F32 direct-conv 路径，S/1 从 `8.16s` 降至 `5.23s`，Mask 从
`13.25s` 降至 `10.29s`。同一完整分割样本的 8-bit alpha 最大差为 `1`、
平均差为 `0.0078`，二值 mask 变化 31 / 1,749,600 像素。可用
`WEVISION_MASK_S1_F32_DIRECT=1` 切回逐像素一致的 F32 审计路径。

Metal 多目标 Mask 共享 encoder、FPN 与 Pixel Fusion，然后按完成的 `<OBJ>`
隐藏特征逐对象执行 transformer 和 alpha cascade，避免 S/1 activation 随目标数
线性增长并触发 GPU OOM。`dogs.jpg / dog` 案例在 32-token 截断前完成 4 个对象，
依次输出 4 组 mask/alpha；峰值 RSS 为 `5.04GB`，Mask 累计耗时 `35.80s`。
Pixel Fusion 从逐批累计 `7.57s` 降至一次 `1.90s`，八张输出图片与未共享版本
逐文件完全一致。检测框如果尚未生成对应 `<OBJ>`，不会伪造 Mask 输出。
