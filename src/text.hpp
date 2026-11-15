#ifndef WEVISION_TEXT_HPP
#define WEVISION_TEXT_HPP

#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace wevision {

struct TextPositionIds {
    int32_t sequence_length = 0;
    // Axis-major [3, sequence_length], matching Qwen's (T,H,W) M-RoPE IDs.
    std::vector<int64_t> values;
};

struct TextRotaryData {
    int32_t sequence_length = 0;
    int32_t head_dim = 0;
    // Token-major [sequence_length, head_dim].
    std::vector<float> cos;
    std::vector<float> sin;
};

struct TextLayerKV {
    ggml_tensor *key = nullptr;   // [head_dim, kv_heads, sequence]
    ggml_tensor *value = nullptr; // [head_dim, kv_heads, sequence]
};

struct TextLayerKVSlot {
    ggml_tensor *key_cache = nullptr;   // [head_dim, kv_heads, capacity]
    ggml_tensor *value_cache = nullptr; // [head_dim, kv_heads, capacity]
    int32_t offset = 0;                 // cached tokens before this step
};

// Computes Qwen3-VL M-RoPE IDs for a single unpadded image request.  Text
// tokens use the same ID on all three axes; image-pad runs use merged T/H/W
// grid coordinates and advance the subsequent text position accordingly.
bool build_text_mrope_positions(const std::vector<int32_t> &input_ids,
                                int32_t image_pad_token_id, int32_t grid_t,
                                int32_t grid_h, int32_t grid_w,
                                int32_t spatial_merge_size,
                                TextPositionIds *output, std::string *error);

bool build_text_rotary_data(const TextPositionIds &positions, int32_t head_dim,
                            float rope_theta,
                            const std::vector<int32_t> &mrope_sections,
                            TextRotaryData *output, std::string *error);

// Builds the first Text Engine boundary: token lookup followed by optional
// replacement of image-pad rows with already-computed Vision embeddings.
bool build_text_input_embeddings(GraphSession &session,
                                 const std::vector<int32_t> &input_ids,
                                 const std::vector<int32_t> &image_positions,
                                 ggml_tensor *vision_embeddings,
                                 ggml_tensor **token_embeddings,
                                 ggml_tensor **output, std::string *error);

bool build_text_rms_norm(GraphSession &session, ggml_tensor *input,
                         const std::string &weight_name, ggml_tensor **output,
                         std::string *error);

bool build_text_linear(GraphSession &session, ggml_tensor *input,
                       const std::string &weight_name, int32_t output_size,
                       ggml_tensor **output, std::string *error);

bool build_text_qk_rope(GraphSession &session, ggml_tensor *query,
                        ggml_tensor *key, const TextRotaryData &rotary,
                        ggml_tensor **query_output, ggml_tensor **key_output,
                        std::string *error);

// Runs one causal grouped-query attention operation. Query/key/value are
// head-major [head_dim, heads, sequence]; the returned tensor keeps the Q
// head layout [head_dim, query_heads, sequence], ready to flatten for o_proj.
bool build_text_causal_attention(GraphSession &session, ggml_tensor *query,
                                 ggml_tensor *key, ggml_tensor *value,
                                 ggml_tensor **output, std::string *error);

// Builds one full Qwen decoder block for prefill. It owns no cache: callers
// provide the complete sequence and use the returned hidden states directly.
bool build_text_decoder_block(GraphSession &session, int32_t layer,
                              ggml_tensor *input, const TextRotaryData &rotary,
                              const TextLayerKV *past_kv,
                              TextLayerKV *kv_output, ggml_tensor **output,
                              std::string *error);

// Executes all decoder blocks for a complete prompt. The returned state is
// pre-output-norm so it matches MaskContext's <OBJ> query boundary; apply
// wv.text.output_norm only on the LM-logits path.
bool build_text_prefill(
    GraphSession &session, ggml_tensor *input, const TextRotaryData &rotary,
    const std::vector<int32_t> &image_positions,
    const std::vector<ggml_tensor *> &deepstack_embeddings,
    std::vector<TextLayerKV> *kv_outputs, ggml_tensor **output,
    std::string *error,
    const std::vector<TextLayerKVSlot> *kv_slots = nullptr);

// Applies Qwen's final RMSNorm and LM head to the last sequence position.
// `last_hidden` is the pre-output-norm state returned by build_text_prefill.
bool build_text_next_logits(GraphSession &session, ggml_tensor *last_hidden,
                            ggml_tensor **output, std::string *error);

// Builds a one-or-more-token incremental decoder graph against an existing
// request KV cache. DeepStack is intentionally prefill-only.
bool build_text_decode(GraphSession &session, ggml_tensor *input,
                       const TextRotaryData &rotary,
                       const std::vector<TextLayerKV> &past_kv,
                       std::vector<TextLayerKV> *new_kv, ggml_tensor **output,
                       std::string *error);

// Incremental decoder that writes new K/V directly into pre-allocated cache
// tensors via ggml_cpy and reads [0, offset + tokens) views for attention,
// avoiding per-step checked_concat copies of the entire history.
bool build_text_decode_inplace(GraphSession &session, ggml_tensor *input,
                               const TextRotaryData &rotary,
                               const std::vector<TextLayerKVSlot> &kv_slots,
                               std::vector<TextLayerKV> *kv_writes,
                               ggml_tensor **output, std::string *error);

} // namespace wevision

#endif
