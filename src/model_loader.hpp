#ifndef WEVISION_MODEL_LOADER_HPP
#define WEVISION_MODEL_LOADER_HPP

#include "wevision.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace wevision {

struct Config {
    uint32_t format_version = 0;

    struct Text {
        uint32_t vocab_size = 0;
        uint32_t hidden_size = 0;
        uint32_t layer_count = 0;
        uint32_t attention_heads = 0;
        uint32_t kv_heads = 0;
        uint32_t head_dim = 0;
        uint32_t intermediate_size = 0;
        uint32_t rope_theta = 0;
        // Optional v1 keys with defaults so pre-eps GGUF files keep loading.
        float rms_norm_eps = 1e-6F;
        std::vector<int32_t> mrope_sections;
    } text;

    struct Vision {
        uint32_t hidden_size = 0;
        uint32_t layer_count = 0;
        uint32_t head_count = 0;
        uint32_t intermediate_size = 0;
        uint32_t patch_size = 0;
        uint32_t temporal_patch_size = 0;
        uint32_t spatial_merge_size = 0;
        uint32_t rope_theta = 10000;
        float layer_norm_eps = 1e-6F;
        std::vector<int32_t> deepstack_layers;
        // Axial position table size (side x side, 48*48 = 2304 for the
        // released checkpoints). Optional key; older GGUF files keep the
        // default, which matches the fixed tensor shape.
        uint32_t position_table_size = 2304;
    } vision;

    struct Mask {
        uint32_t seg_token_id = 0;
        uint32_t spatial_merge_size = 0;
        uint32_t hidden_size = 0;
        uint32_t feature_size = 0;
        uint32_t msda_levels = 0;
        uint32_t msda_heads = 0;
        uint32_t msda_points = 0;
        uint32_t transformer_layers = 0;
        uint32_t cnn_image_size = 0;
        // Optional v1 keys with defaults for older GGUF files.
        uint32_t encoder_layers = 3;
        uint32_t norm_groups = 32;
        float norm_eps = 1e-5F;
        std::vector<int32_t> vision_feature_layers;
    } mask;
};

// Raw Hugging Face tokenizer resources stored in GGUF. Parsing and BPE
// execution belong to the text module; Model only owns immutable assets.
struct TokenizerAssets {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template;
};

struct Model {
    Model() = default;
    ~Model();

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Config config;
    TokenizerAssets tokenizer;
    ::gguf_context *gguf = nullptr;
    ::ggml_context *metadata = nullptr;
    std::unordered_map<std::string, ::ggml_tensor *> tensors;
    std::string path;
    std::string last_error;
};

std::unique_ptr<Model> load_model(const std::string &path,
                                  wevision_status &status);
wevision_status validate_model(const Model &model, std::string *error);

} // namespace wevision

#endif
