#ifndef WEVISION_VISION_HPP
#define WEVISION_VISION_HPP

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

namespace wevision {

// Interleaved RGB8 image supplied by the embedding application. File decoding
// is intentionally outside this module so the public C API can stay free of a
// PNG/JPEG dependency.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // H * W * 3, RGB order
};

struct VisionPreprocessConfig {
    int patch_size = 16;
    int temporal_patch_size = 2;
    int spatial_merge_size = 2;
    int min_pixels = 16 * 32 * 32;
    int max_pixels = 2025 * 32 * 32;
    float image_mean[3] = {0.5F, 0.5F, 0.5F};
    float image_std[3] = {0.5F, 0.5F, 0.5F};
};

struct SmartResizeShape {
    int height = 0;
    int width = 0;
};

struct PatchifiedImage {
    int grid_t = 1;
    int grid_h = 0;
    int grid_w = 0;
    int patch_dim = 0;
    // Row-major [grid_t * grid_h * grid_w, patch_dim]. Patch order matches
    // Qwen2/3-VL processor: [block_y, block_x, intra_y, intra_x, C, T, Y, X].
    std::vector<float> values;
};

struct VisionImageInput {
    PatchifiedImage patches;
    // Equivalent to one row of Qwen's image_grid_thw: [T, H, W].
    int grid_thw[3] = {1, 0, 0};
};

struct MaskImageInput {
    int original_width = 0;
    int original_height = 0;
    int width = 0;
    int height = 0;
    // Planar RGB float32 [3, height, width] in the [0, 1] range.
    std::vector<float> rgb;
};

// CPU-side, deterministic description of the vision position embedding lookup.
// Every vector is token-major and uses the encoder's merge-block token order.
struct VisionPositionData {
    int token_count = 0;
    std::vector<int32_t> axial_hw;          // [tokens, 2], H then W
    std::vector<int32_t> interpolation_ids; // [tokens, 4], flattened table IDs
    std::vector<float> interpolation_weights; // [tokens, 4], sums to one
};

struct VisionRotaryData {
    int token_count = 0;
    int head_dim = 0;
    std::vector<float> cos; // [tokens, head_dim]
    std::vector<float> sin; // [tokens, head_dim]
};

struct VisionAttentionQKV {
    ggml_tensor *query = nullptr; // [head_dim, heads, tokens]
    ggml_tensor *key = nullptr;
    ggml_tensor *value = nullptr;
};

class GraphSession;

// Conv3D patch embedding is a projection over the processor's packed patches.
// Higher vision stages intentionally remain separate for traceable alignment.
bool build_vision_patch_embed(GraphSession &session,
                              const PatchifiedImage &input,
                              ggml_tensor **output, std::string *error);

// Adds the dynamically interpolated learned position embedding to patch output.
// `patch_embedding` has ggml shape [hidden, tokens].
bool build_vision_positioned_input(GraphSession &session,
                                   ggml_tensor *patch_embedding,
                                   const VisionPositionData &positions,
                                   ggml_tensor **output, std::string *error);

// LayerNorm used by Qwen vision blocks (eps=1e-6), including affine scale/bias.
bool build_vision_layer_norm(GraphSession &session, ggml_tensor *input,
                             const std::string &weight_name,
                             const std::string &bias_name, ggml_tensor **output,
                             std::string *error);

bool build_vision_linear(GraphSession &session, ggml_tensor *input,
                         const std::string &weight_name,
                         const std::string &bias_name, int output_size,
                         ggml_tensor **output, std::string *error);

bool build_vision_qkv_rotary(GraphSession &session, ggml_tensor *packed_qkv,
                             const VisionRotaryData &rotary,
                             VisionAttentionQKV *output, std::string *error);

bool build_vision_attention(GraphSession &session,
                            const VisionAttentionQKV &qkv, ggml_tensor **output,
                            std::string *error);

bool build_vision_block(GraphSession &session, int layer, ggml_tensor *input,
                        const VisionRotaryData &rotary, ggml_tensor **output,
                        std::string *error);

bool build_vision_merger(GraphSession &session, ggml_tensor *input,
                         const std::string &prefix, bool norm_after_shuffle,
                         ggml_tensor **output, std::string *error);

// Equivalent to Transformers Qwen2VLImageProcessor.smart_resize(). It only
// computes the target size; callers perform bicubic resize before patchifying.
bool smart_resize(int source_height, int source_width,
                  const VisionPreprocessConfig &config,
                  SmartResizeShape *output);

// Normalizes and packs an image that has already been resized to the smart
// shape. The strict shape check prevents accidental preprocessing with a
// visually plausible but incompatible canvas.
bool patchify_resized_rgb(const RgbImage &image,
                          const VisionPreprocessConfig &config,
                          PatchifiedImage *output);

// Resize source RGB8 with the same cubic kernel convention used by Pillow,
// then normalize and pack it for the Qwen vision encoder.
bool prepare_vision_image(const RgbImage &source,
                          const VisionPreprocessConfig &config,
                          VisionImageInput *output);

// Independently resize the original image to the square canvas consumed by
// the mask decoder. This intentionally does not reuse Qwen smart-resize.
bool prepare_mask_image(const RgbImage &source, int cnn_image_size,
                        MaskImageInput *output);

// Axial learned-position table side: 48 x 48 = 2304 positions, fixed by the
// released checkpoint's pos_embd tensor shape.
constexpr int kVisionPositionTableSide = 48;

// Reproduces Qwen3-VL's block-major 2-D ids and bilinear learned-position
// table lookup (align_corners=true). `position_table_side` is 48 for the
// released checkpoint (2304 learned positions).
bool build_vision_position_data(int grid_h, int grid_w, int spatial_merge_size,
                                int position_table_side,
                                VisionPositionData *output);

// Qwen3-VL axial 2-D RoPE table. `head_dim` must be divisible by four.
bool build_vision_rotary_data(const VisionPositionData &positions, int head_dim,
                              float rope_theta, VisionRotaryData *output);

} // namespace wevision

#endif
