#include "model_loader.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <filesystem>
#include <sstream>
#include <utility>

namespace wevision {
namespace {

constexpr const char *kArchitecture = "wevision.qwen3vl";
constexpr uint32_t kFormatVersion = 1;

bool get_u32(gguf_context *gguf, const char *key, uint32_t *value) {
    const int64_t index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_UINT32) {
        return false;
    }
    *value = gguf_get_val_u32(gguf, index);
    return true;
}

bool get_f32(gguf_context *gguf, const char *key, float *value) {
    const int64_t index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    *value = gguf_get_val_f32(gguf, index);
    return true;
}

bool get_string(gguf_context *gguf, const char *key, std::string *value) {
    const int64_t index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_STRING) {
        return false;
    }
    *value = gguf_get_val_str(gguf, index);
    return true;
}

bool get_i32_array(gguf_context *gguf, const char *key,
                   std::vector<int32_t> *value) {
    const int64_t index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf, index) != GGUF_TYPE_INT32) {
        return false;
    }
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto *data =
        static_cast<const int32_t *>(gguf_get_arr_data(gguf, index));
    value->assign(data, data + count);
    return true;
}

std::string text_block_name(uint32_t layer, const char *suffix) {
    return "wv.text.blk." + std::to_string(layer) + "." + suffix;
}

std::string vision_block_name(uint32_t layer, const char *suffix) {
    return "wv.vision.blk." + std::to_string(layer) + "." + suffix;
}

bool expect_tensor(const Model &model, const std::string &name, ggml_type type,
                   const std::vector<int64_t> &shape, std::string *error) {
    const auto found = model.tensors.find(name);
    if (found == model.tensors.end()) {
        *error = "missing tensor: " + name;
        return false;
    }
    const ggml_tensor *tensor = found->second;
    if (tensor->type != type) {
        // Offline quantization may replace bf16 linear weights with block
        // quantized variants (wevision-quantize q8_0/q4_k). Vectors such as
        // norms and biases are never quantized, so they still match exactly.
        // An f32 conversion pipeline (e.g. the 4B build converted with
        // --f32, whose kept tensors survive quantization untouched) stores
        // the same weights losslessly at full width; f32 runs natively in
        // every compute path, so accept it alongside the bf16 layout.
        const bool quantized_variant =
            type == GGML_TYPE_BF16 &&
            (tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q4_K ||
             tensor->type == GGML_TYPE_F32);
        if (!quantized_variant) {
            *error = "unexpected type for " + name;
            return false;
        }
    }
    if (ggml_n_dims(tensor) != static_cast<int>(shape.size())) {
        *error = "unexpected rank for " + name;
        return false;
    }
    size_t index = 0;
    for (const int64_t expected : shape) {
        if (tensor->ne[index++] != expected) {
            std::ostringstream message;
            message << "unexpected shape for " << name << ": got [";
            for (int dim = 0; dim < ggml_n_dims(tensor); ++dim) {
                if (dim)
                    message << ", ";
                message << tensor->ne[dim];
            }
            message << "]";
            *error = message.str();
            return false;
        }
    }
    return true;
}

bool expect_vector(const Model &model, const std::string &name, ggml_type type,
                   int64_t length, std::string *error) {
    return expect_tensor(model, name, type, {length}, error);
}

bool load_config(gguf_context *gguf, Config *config, std::string *error) {
    std::string architecture;
    if (!get_string(gguf, "general.architecture", &architecture) ||
        architecture != kArchitecture) {
        *error = "general.architecture must be wevision.qwen3vl";
        return false;
    }
    if (!get_u32(gguf, "wevision.format_version", &config->format_version) ||
        config->format_version != kFormatVersion) {
        *error = "unsupported wevision.format_version";
        return false;
    }
    auto &text = config->text;
    if (!get_u32(gguf, "wv.text.vocab_size", &text.vocab_size) ||
        !get_u32(gguf, "wv.text.hidden_size", &text.hidden_size) ||
        !get_u32(gguf, "wv.text.layer_count", &text.layer_count) ||
        !get_u32(gguf, "wv.text.attention_head_count", &text.attention_heads) ||
        !get_u32(gguf, "wv.text.attention_head_count_kv", &text.kv_heads) ||
        !get_u32(gguf, "wv.text.head_dim", &text.head_dim) ||
        !get_u32(gguf, "wv.text.intermediate_size", &text.intermediate_size) ||
        !get_u32(gguf, "wv.text.rope_theta", &text.rope_theta) ||
        !get_i32_array(gguf, "wv.text.mrope_sections", &text.mrope_sections)) {
        *error = "incomplete wv.text metadata";
        return false;
    }
    auto &vision = config->vision;
    if (!get_u32(gguf, "wv.vision.hidden_size", &vision.hidden_size) ||
        !get_u32(gguf, "wv.vision.layer_count", &vision.layer_count) ||
        !get_u32(gguf, "wv.vision.head_count", &vision.head_count) ||
        !get_u32(gguf, "wv.vision.intermediate_size",
                 &vision.intermediate_size) ||
        !get_u32(gguf, "wv.vision.patch_size", &vision.patch_size) ||
        !get_u32(gguf, "wv.vision.temporal_patch_size",
                 &vision.temporal_patch_size) ||
        !get_u32(gguf, "wv.vision.spatial_merge_size",
                 &vision.spatial_merge_size) ||
        !get_i32_array(gguf, "wv.vision.deepstack_layers",
                       &vision.deepstack_layers)) {
        *error = "incomplete wv.vision metadata";
        return false;
    }
    auto &mask = config->mask;
    if (!get_u32(gguf, "wv.mask.seg_token_id", &mask.seg_token_id) ||
        !get_u32(gguf, "wv.mask.spatial_merge_size",
                 &mask.spatial_merge_size) ||
        !get_u32(gguf, "wv.mask.hidden_size", &mask.hidden_size) ||
        !get_u32(gguf, "wv.mask.feature_size", &mask.feature_size) ||
        !get_u32(gguf, "wv.mask.msda_levels", &mask.msda_levels) ||
        !get_u32(gguf, "wv.mask.msda_heads", &mask.msda_heads) ||
        !get_u32(gguf, "wv.mask.msda_points", &mask.msda_points) ||
        !get_u32(gguf, "wv.mask.transformer_layers",
                 &mask.transformer_layers) ||
        !get_u32(gguf, "wv.mask.cnn_image_size", &mask.cnn_image_size) ||
        !get_i32_array(gguf, "wv.mask.vision_feature_layers",
                       &mask.vision_feature_layers)) {
        *error = "incomplete wv.mask metadata";
        return false;
    }
    // Optional v1 keys written by newer converters; older GGUF files keep
    // the struct defaults, which match the reference implementation.
    get_f32(gguf, "wv.text.rms_norm_eps", &text.rms_norm_eps);
    get_u32(gguf, "wv.vision.rope_theta", &vision.rope_theta);
    get_u32(gguf, "wv.vision.position_table_size", &vision.position_table_size);
    get_f32(gguf, "wv.vision.layer_norm_eps", &vision.layer_norm_eps);
    get_u32(gguf, "wv.mask.encoder_layers", &mask.encoder_layers);
    get_u32(gguf, "wv.mask.norm_groups", &mask.norm_groups);
    get_f32(gguf, "wv.mask.norm_eps", &mask.norm_eps);
    const uint64_t table_side = static_cast<uint64_t>(
        std::sqrt(static_cast<double>(vision.position_table_size)) + 0.5);
    if (table_side * table_side != vision.position_table_size) {
        *error = "wv.vision.position_table_size must be a perfect square";
        return false;
    }
    // Qwen3-style text blocks decouple head_dim from hidden_size (the 4B
    // build: hidden 2560 vs 32*128 = 4096 attention width, with o_proj
    // projecting back down), so only validate the attention geometry here.
    if (text.head_dim <= 0 || text.head_dim % 2 != 0 ||
        text.attention_heads == 0 || text.kv_heads == 0 ||
        text.attention_heads % text.kv_heads != 0 ||
        vision.hidden_size % vision.head_count != 0 ||
        mask.hidden_size % mask.msda_heads != 0 ||
        mask.msda_levels != mask.vision_feature_layers.size()) {
        *error = "inconsistent model dimensions";
        return false;
    }
    return true;
}

bool load_tokenizer_assets(gguf_context *gguf, TokenizerAssets *assets,
                           std::string *error) {
    if (!get_string(gguf, "wv.hf.tokenizer_json", &assets->tokenizer_json) ||
        !get_string(gguf, "wv.hf.tokenizer_config_json",
                    &assets->tokenizer_config_json) ||
        !get_string(gguf, "wv.hf.chat_template", &assets->chat_template) ||
        assets->tokenizer_json.empty() ||
        assets->tokenizer_config_json.empty() ||
        assets->chat_template.empty()) {
        *error = "incomplete tokenizer assets in GGUF metadata";
        return false;
    }
    return true;
}

} // namespace

Model::~Model() {
    if (gguf != nullptr)
        gguf_free(gguf);
    if (metadata != nullptr)
        ggml_free(metadata);
}

std::unique_ptr<Model> load_model(const std::string &path,
                                  wevision_status &status) {
    status = WEVISION_ERROR_INTERNAL;
    if (path.empty()) {
        status = WEVISION_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!std::filesystem::is_regular_file(path)) {
        status = WEVISION_ERROR_FILE_NOT_FOUND;
        return nullptr;
    }

    auto model = std::make_unique<Model>();
    model->path = path;
    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/&model->metadata};
    model->gguf = gguf_init_from_file(path.c_str(), params);
    if (model->gguf == nullptr || model->metadata == nullptr) {
        status = WEVISION_ERROR_MODEL_FORMAT;
        return nullptr;
    }

    std::string error;
    if (!load_config(model->gguf, &model->config, &error)) {
        model->last_error = std::move(error);
        status = WEVISION_ERROR_MODEL_FORMAT;
        return model;
    }
    if (!load_tokenizer_assets(model->gguf, &model->tokenizer, &error)) {
        model->last_error = std::move(error);
        status = WEVISION_ERROR_MODEL_FORMAT;
        return model;
    }

    const int64_t tensor_count = gguf_get_n_tensors(model->gguf);
    for (int64_t index = 0; index < tensor_count; ++index) {
        const char *name = gguf_get_tensor_name(model->gguf, index);
        if (name == nullptr) {
            model->last_error = "invalid GGUF tensor descriptor";
            status = WEVISION_ERROR_MODEL_FORMAT;
            return model;
        }
        ggml_tensor *tensor = ggml_get_tensor(model->metadata, name);
        if (tensor == nullptr || !model->tensors.emplace(name, tensor).second) {
            model->last_error = "invalid or duplicate GGUF tensor descriptor";
            status = WEVISION_ERROR_MODEL_FORMAT;
            return model;
        }
    }

    const wevision_status validation =
        validate_model(*model, &model->last_error);
    status = validation;
    return model;
}

wevision_status validate_model(const Model &model, std::string *error) {
    const auto &text = model.config.text;
    const auto &vision = model.config.vision;
    const auto &mask = model.config.mask;
    // The inventory is validated name-by-name below and by the graph
    // builders; a fixed total would pin the loader to one released layout
    // (the 2B build carries 626+296 tensors, other family sizes differ).
    // Only reject tensors outside the wv.* namespace.
    for (const auto &entry : model.tensors) {
        if (entry.first.rfind("wv.", 0) != 0) {
            *error = "unexpected tensor name outside the wv.* namespace: " +
                     entry.first;
            return WEVISION_ERROR_MODEL_FORMAT;
        }
    }
    if (!expect_tensor(model, "wv.text.token_embd.weight", GGML_TYPE_BF16,
                       {text.hidden_size, text.vocab_size}, error) ||
        !expect_tensor(model, "wv.text.output.weight", GGML_TYPE_BF16,
                       {text.hidden_size, text.vocab_size}, error) ||
        !expect_vector(model, "wv.text.output_norm.weight", GGML_TYPE_BF16,
                       text.hidden_size, error)) {
        return WEVISION_ERROR_MODEL_FORMAT;
    }
    for (uint32_t layer = 0; layer < text.layer_count; ++layer) {
        const int64_t kv_dim =
            static_cast<int64_t>(text.kv_heads) * text.head_dim;
        const std::vector<std::pair<const char *, std::vector<int64_t>>>
            matrices = {
                // Qwen3-style blocks decouple the attention width from
                // hidden_size (the 4B build: q [in=2560, out=4096], o
                // [in=4096, out=2560]); the 2B build is symmetric, so the
                // old hidden_size x hidden_size expectation held by accident.
                {"attn_q.weight",
                 {text.hidden_size, text.attention_heads * text.head_dim}},
                {"attn_k.weight", {text.hidden_size, kv_dim}},
                {"attn_v.weight", {text.hidden_size, kv_dim}},
                {"attn_output.weight",
                 {text.attention_heads * text.head_dim, text.hidden_size}},
                {"ffn_gate.weight", {text.hidden_size, text.intermediate_size}},
                {"ffn_up.weight", {text.hidden_size, text.intermediate_size}},
                {"ffn_down.weight", {text.intermediate_size, text.hidden_size}},
            };
        for (const auto &item : matrices) {
            if (!expect_tensor(model, text_block_name(layer, item.first),
                               GGML_TYPE_BF16, item.second, error)) {
                return WEVISION_ERROR_MODEL_FORMAT;
            }
        }
        for (const char *suffix : {"attn_norm.weight", "ffn_norm.weight"}) {
            if (!expect_vector(model, text_block_name(layer, suffix),
                               GGML_TYPE_BF16, text.hidden_size, error)) {
                return WEVISION_ERROR_MODEL_FORMAT;
            }
        }
        for (const char *suffix :
             {"attn_q_norm.weight", "attn_k_norm.weight"}) {
            if (!expect_vector(model, text_block_name(layer, suffix),
                               GGML_TYPE_BF16, text.head_dim, error)) {
                return WEVISION_ERROR_MODEL_FORMAT;
            }
        }
    }

    if (!expect_tensor(model, "wv.vision.patch_embed.weight", GGML_TYPE_BF16,
                       {vision.patch_size, vision.patch_size,
                        3 * vision.temporal_patch_size, vision.hidden_size},
                       error) ||
        !expect_vector(model, "wv.vision.patch_embed.bias", GGML_TYPE_BF16,
                       vision.hidden_size, error) ||
        !expect_tensor(model, "wv.vision.pos_embd.weight", GGML_TYPE_BF16,
                       {vision.hidden_size,
                        static_cast<int64_t>(vision.position_table_size)},
                       error)) {
        return WEVISION_ERROR_MODEL_FORMAT;
    }
    for (uint32_t layer = 0; layer < vision.layer_count; ++layer) {
        if (!expect_tensor(model, vision_block_name(layer, "attn_qkv.weight"),
                           GGML_TYPE_BF16,
                           {vision.hidden_size, 3 * vision.hidden_size},
                           error) ||
            !expect_tensor(model,
                           vision_block_name(layer, "attn_output.weight"),
                           GGML_TYPE_BF16,
                           {vision.hidden_size, vision.hidden_size}, error) ||
            !expect_tensor(model, vision_block_name(layer, "ffn_up.weight"),
                           GGML_TYPE_BF16,
                           {vision.hidden_size, vision.intermediate_size},
                           error) ||
            !expect_tensor(model, vision_block_name(layer, "ffn_down.weight"),
                           GGML_TYPE_BF16,
                           {vision.intermediate_size, vision.hidden_size},
                           error)) {
            return WEVISION_ERROR_MODEL_FORMAT;
        }
    }

    if (!expect_tensor(model, "wv.mask.query_builder.content_proj.weight",
                       GGML_TYPE_F32, {text.hidden_size, mask.hidden_size},
                       error) ||
        !expect_tensor(model, "wv.mask.visual_encoder.level_embed",
                       GGML_TYPE_F32, {mask.hidden_size, mask.msda_levels},
                       error)) {
        return WEVISION_ERROR_MODEL_FORMAT;
    }
    for (uint32_t layer = 0; layer < mask.transformer_layers; ++layer) {
        const std::string prefix =
            "wv.mask.decoder.blk." + std::to_string(layer) + ".cross_attn.";
        if (!expect_tensor(
                model, prefix + "sampling_offsets.weight", GGML_TYPE_F32,
                {mask.hidden_size,
                 mask.msda_heads * mask.msda_levels * mask.msda_points * 2},
                error) ||
            !expect_tensor(
                model, prefix + "attention_weights.weight", GGML_TYPE_F32,
                {mask.hidden_size,
                 mask.msda_heads * mask.msda_levels * mask.msda_points},
                error) ||
            !expect_tensor(model, prefix + "value_proj.weight", GGML_TYPE_F32,
                           {mask.hidden_size, mask.hidden_size}, error)) {
            return WEVISION_ERROR_MODEL_FORMAT;
        }
    }
    return WEVISION_OK;
}

} // namespace wevision
