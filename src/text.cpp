#include "text.hpp"

#include "model_loader.hpp"

#include "graph_ops.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace wevision {

bool build_text_mrope_positions(const std::vector<int32_t> &input_ids,
                                int32_t image_pad_token_id, int32_t grid_t,
                                int32_t grid_h, int32_t grid_w,
                                int32_t spatial_merge_size,
                                TextPositionIds *output, std::string *error) {
    if (output == nullptr || input_ids.empty() || image_pad_token_id < 0 ||
        grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || spatial_merge_size <= 0 ||
        grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
        set_error(error, "invalid Qwen M-RoPE position input");
        return false;
    }
    const int32_t merged_h = grid_h / spatial_merge_size;
    const int32_t merged_w = grid_w / spatial_merge_size;
    const int32_t image_tokens = grid_t * merged_h * merged_w;
    TextPositionIds result;
    result.sequence_length = static_cast<int32_t>(input_ids.size());
    result.values.assign(static_cast<size_t>(result.sequence_length) * 3U, 0);
    int64_t current_position = 0;
    for (size_t start = 0; start < input_ids.size();) {
        const bool is_image = input_ids[start] == image_pad_token_id;
        size_t end = start + 1;
        while (end < input_ids.size() &&
               (input_ids[end] == image_pad_token_id) == is_image)
            ++end;
        const size_t count = end - start;
        if (!is_image) {
            for (size_t index = start; index < end; ++index)
                for (int axis = 0; axis < 3; ++axis) {
                    result.values[static_cast<size_t>(axis) * input_ids.size() +
                                  index] =
                        current_position + static_cast<int64_t>(index - start);
                }
            current_position += static_cast<int64_t>(count);
        } else {
            if (count != static_cast<size_t>(image_tokens)) {
                set_error(error,
                          "image-pad run does not match merged image grid");
                return false;
            }
            size_t index = start;
            for (int32_t time = 0; time < grid_t; ++time)
                for (int32_t row = 0; row < merged_h; ++row)
                    for (int32_t col = 0; col < merged_w; ++col, ++index) {
                        result.values[index] = current_position + time;
                        result.values[input_ids.size() + index] =
                            current_position + row;
                        result.values[2U * input_ids.size() + index] =
                            current_position + col;
                    }
            current_position += std::max(merged_h, merged_w);
        }
        start = end;
    }
    *output = std::move(result);
    return true;
}

bool build_text_rotary_data(const TextPositionIds &positions, int32_t head_dim,
                            float rope_theta,
                            const std::vector<int32_t> &mrope_sections,
                            TextRotaryData *output, std::string *error) {
    if (output == nullptr || positions.sequence_length <= 0 ||
        positions.values.size() !=
            static_cast<size_t>(positions.sequence_length) * 3U ||
        head_dim <= 0 || head_dim % 2 != 0 || rope_theta <= 1.0F ||
        mrope_sections.size() != 3U || mrope_sections[0] < 0 ||
        mrope_sections[1] < 0 || mrope_sections[2] < 0) {
        set_error(error, "invalid text rotary input");
        return false;
    }
    TextRotaryData result;
    result.sequence_length = positions.sequence_length;
    result.head_dim = head_dim;
    result.cos.resize(static_cast<size_t>(positions.sequence_length) *
                      head_dim);
    result.sin.resize(result.cos.size());
    const int32_t half_dim = head_dim / 2;
    for (int32_t token = 0; token < positions.sequence_length; ++token) {
        for (int32_t dim = 0; dim < half_dim; ++dim) {
            int axis = 0;
            if (dim < mrope_sections[1] * 3 && dim % 3 == 1)
                axis = 1;
            if (dim < mrope_sections[2] * 3 && dim % 3 == 2)
                axis = 2;
            const float inv_freq =
                1.0F / std::pow(rope_theta, static_cast<float>(2 * dim) /
                                                static_cast<float>(head_dim));
            const float angle =
                static_cast<float>(
                    positions.values[static_cast<size_t>(axis) *
                                         positions.sequence_length +
                                     token]) *
                inv_freq;
            const size_t base = static_cast<size_t>(token) * head_dim;
            result.cos[base + dim] = result.cos[base + half_dim + dim] =
                std::cos(angle);
            result.sin[base + dim] = result.sin[base + half_dim + dim] =
                std::sin(angle);
        }
    }
    *output = std::move(result);
    return true;
}

bool build_text_rms_norm(GraphSession &session, ggml_tensor *input,
                         const std::string &weight_name, ggml_tensor **output,
                         std::string *error) {
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] <= 0) {
        set_error(error, "invalid text RMSNorm input");
        return false;
    }
    ggml_tensor *weight = session.weight(weight_name, error);
    if (weight == nullptr)
        return false;
    if (ggml_nelements(weight) != input->ne[0]) {
        set_error(error, "invalid text RMSNorm weight shape: " + weight_name);
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *normalized =
        ggml_rms_norm(context, input, session.model().config.text.rms_norm_eps);
    ggml_tensor *scale = ggml_cast(context, weight, GGML_TYPE_F32);
    *output = checked_mul(context, normalized,
                          ggml_reshape_2d(context, scale, input->ne[0], 1));
    return true;
}

bool build_text_qk_rope(GraphSession &session, ggml_tensor *query,
                        ggml_tensor *key, const TextRotaryData &rotary,
                        ggml_tensor **query_output, ggml_tensor **key_output,
                        std::string *error) {
    if (query == nullptr || key == nullptr || query_output == nullptr ||
        key_output == nullptr || query->type != GGML_TYPE_F32 ||
        key->type != GGML_TYPE_F32 || query->ne[0] != rotary.head_dim ||
        key->ne[0] != rotary.head_dim ||
        query->ne[2] != rotary.sequence_length ||
        key->ne[2] != rotary.sequence_length ||
        rotary.cos.size() !=
            static_cast<size_t>(rotary.sequence_length) * rotary.head_dim ||
        rotary.sin.size() != rotary.cos.size()) {
        set_error(error, "invalid text Q/K rotary input");
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *cos = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, rotary.head_dim, rotary.sequence_length);
    ggml_tensor *sin = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, rotary.head_dim, rotary.sequence_length);
    std::vector<float> cos_data(rotary.cos.size()), sin_data(rotary.sin.size());
    for (int32_t token = 0; token < rotary.sequence_length; ++token)
        for (int32_t dim = 0; dim < rotary.head_dim; ++dim) {
            cos_data[static_cast<size_t>(dim) +
                     static_cast<size_t>(rotary.head_dim) * token] =
                rotary.cos[static_cast<size_t>(token) * rotary.head_dim + dim];
            sin_data[static_cast<size_t>(dim) +
                     static_cast<size_t>(rotary.head_dim) * token] =
                rotary.sin[static_cast<size_t>(token) * rotary.head_dim + dim];
        }
    if (!session.set_input(cos, cos_data.data(), ggml_nbytes(cos), error) ||
        !session.set_input(sin, sin_data.data(), ggml_nbytes(sin), error))
        return false;
    auto rotate = [context, &rotary](ggml_tensor *value) {
        const int32_t half = rotary.head_dim / 2;
        ggml_tensor *first =
            ggml_view_3d(context, value, half, value->ne[1], value->ne[2],
                         value->nb[1], value->nb[2], 0);
        ggml_tensor *second = ggml_view_3d(
            context, value, half, value->ne[1], value->ne[2], value->nb[1],
            value->nb[2], static_cast<size_t>(half) * sizeof(float));
        return checked_concat(context, ggml_neg(context, second), first, 0);
    };
    auto apply = [context, &rotary, cos, sin, &rotate](ggml_tensor *value) {
        ggml_tensor *c =
            checked_repeat(context,
                           ggml_reshape_3d(context, cos, rotary.head_dim, 1,
                                           rotary.sequence_length),
                           value);
        ggml_tensor *s =
            checked_repeat(context,
                           ggml_reshape_3d(context, sin, rotary.head_dim, 1,
                                           rotary.sequence_length),
                           value);
        return checked_add(context, checked_mul(context, value, c),
                           checked_mul(context, rotate(value), s));
    };
    *query_output = apply(query);
    *key_output = apply(key);
    return true;
}

bool build_text_causal_attention(GraphSession &session, ggml_tensor *query,
                                 ggml_tensor *key, ggml_tensor *value,
                                 ggml_tensor **output, std::string *error) {
    if (query == nullptr || key == nullptr || value == nullptr ||
        output == nullptr || query->type != GGML_TYPE_F32 ||
        (key->type != GGML_TYPE_F32 && key->type != GGML_TYPE_F16) ||
        value->type != key->type ||
        query->ne[0] <= 0 || query->ne[0] != key->ne[0] ||
        query->ne[0] != value->ne[0] || query->ne[1] <= 0 || key->ne[1] <= 0 ||
        key->ne[1] != value->ne[1] || query->ne[1] % key->ne[1] != 0 ||
        query->ne[2] <= 0 || key->ne[2] < query->ne[2] ||
        key->ne[2] != value->ne[2]) {
        set_error(error, "invalid text causal attention input");
        return false;
    }
    ggml_context *context = session.context();
    const int64_t query_tokens = query->ne[2];
    const int64_t key_tokens = key->ne[2];
    // ggml's mask layout is [key_position, query_position, heads, batch].
    // It must be FP16; zero keeps a score and -inf makes future keys causal.
    std::vector<ggml_fp16_t> mask_data(
        static_cast<size_t>(key_tokens * query_tokens));
    const int64_t query_offset = key_tokens - query_tokens;
    for (int64_t position = 0; position < query_tokens; ++position) {
        for (int64_t key_position = 0; key_position < key_tokens;
             ++key_position) {
            mask_data[static_cast<size_t>(key_position +
                                          key_tokens * position)] =
                ggml_fp32_to_fp16(
                    key_position <= query_offset + position ? 0.0F : -INFINITY);
        }
    }
    ggml_tensor *mask =
        ggml_new_tensor_2d(context, GGML_TYPE_F16, key_tokens, query_tokens);
    if (!session.set_input(mask, mask_data.data(), ggml_nbytes(mask), error))
        return false;

    // [D,H,T] -> [D,T,H]. ggml's fused kernel broadcasts KV heads over Q
    // heads, which is precisely Qwen's grouped-query attention, and accepts
    // non-contiguous permuted views for K and V directly.
    ggml_tensor *q =
        ggml_cont(context, ggml_permute(context, query, 0, 2, 1, 3));
    ggml_tensor *k = ggml_permute(context, key, 0, 2, 1, 3);
    ggml_tensor *v = ggml_permute(context, value, 0, 2, 1, 3);
    *output = ggml_flash_attn_ext(
        context, q, k, v, mask,
        1.0F / std::sqrt(static_cast<float>(query->ne[0])), 0.0F, 0.0F);
    return true;
}

namespace {

bool build_text_decoder_block_impl(GraphSession &session, int32_t layer,
                                   ggml_tensor *input,
                                   const TextRotaryData &rotary,
                                   const TextLayerKV *past_kv,
                                   const TextLayerKVSlot *kv_slot,
                                   bool read_kv_from_slot,
                                   TextLayerKV *kv_output, ggml_tensor **output,
                                   std::string *error) {
    const auto &config = session.model().config;
    if (layer < 0 || static_cast<uint32_t>(layer) >= config.text.layer_count ||
        input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] != config.text.hidden_size ||
        input->ne[1] != rotary.sequence_length) {
        set_error(error, "invalid text decoder block input");
        return false;
    }
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "wv.text.blk.%d.", layer);
    const std::string name(prefix);
    auto checkpoint = [&](const char *suffix, ggml_tensor *tensor,
                          const char *layout) {
        char checkpoint_name[80];
        std::snprintf(checkpoint_name, sizeof(checkpoint_name),
                      "text.block.%02d.%s", layer, suffix);
        return session.checkpoint({checkpoint_name, "text", layout}, tensor,
                                  error);
    };
    ggml_tensor *norm = nullptr;
    ggml_tensor *q = nullptr;
    ggml_tensor *k = nullptr;
    ggml_tensor *v = nullptr;
    if (!build_text_rms_norm(session, input, name + "attn_norm.weight", &norm,
                             error) ||
        !build_text_linear(session, norm, name + "attn_q.weight",
                           config.text.attention_heads * config.text.head_dim,
                           &q, error) ||
        !build_text_linear(session, norm, name + "attn_k.weight",
                           config.text.kv_heads * config.text.head_dim, &k,
                           error) ||
        !build_text_linear(session, norm, name + "attn_v.weight",
                           config.text.kv_heads * config.text.head_dim, &v,
                           error))
        return false;
    if (!checkpoint("norm", norm, "B,L,H") || !checkpoint("q", q, "B,L,QH") ||
        !checkpoint("k", k, "B,L,KVH") || !checkpoint("v", v, "B,L,KVH"))
        return false;

    const int64_t tokens = input->ne[1];
    q = ggml_reshape_3d(session.context(), q, config.text.head_dim,
                        config.text.attention_heads, tokens);
    k = ggml_reshape_3d(session.context(), k, config.text.head_dim,
                        config.text.kv_heads, tokens);
    ggml_tensor *q_norm = nullptr;
    ggml_tensor *k_norm = nullptr;
    if (!build_text_rms_norm(session, q, name + "attn_q_norm.weight", &q_norm,
                             error) ||
        !build_text_rms_norm(session, k, name + "attn_k_norm.weight", &k_norm,
                             error))
        return false;
    if (!checkpoint("q_norm", q_norm, "B,L,heads,head_dim") ||
        !checkpoint("k_norm", k_norm, "B,L,kv_heads,head_dim"))
        return false;
    ggml_tensor *q_rope = nullptr;
    ggml_tensor *k_rope = nullptr;
    if (!build_text_qk_rope(session, q_norm, k_norm, rotary, &q_rope, &k_rope,
                            error))
        return false;
    if (!checkpoint("q_rope", q_rope, "B,L,heads,head_dim") ||
        !checkpoint("k_rope", k_rope, "B,L,kv_heads,head_dim"))
        return false;
    ggml_tensor *v_heads =
        ggml_reshape_3d(session.context(), v, config.text.head_dim,
                        config.text.kv_heads, tokens);
    if (kv_output != nullptr)
        *kv_output = {k_rope, v_heads};
    ggml_tensor *attention_key = k_rope;
    ggml_tensor *attention_value = v_heads;
    if (kv_slot != nullptr) {
        if (kv_slot->key_cache == nullptr || kv_slot->value_cache == nullptr ||
            (kv_slot->key_cache->type != GGML_TYPE_F32 &&
             kv_slot->key_cache->type != GGML_TYPE_F16) ||
            kv_slot->value_cache->type != kv_slot->key_cache->type ||
            kv_slot->key_cache->ne[0] != config.text.head_dim ||
            kv_slot->value_cache->ne[0] != config.text.head_dim ||
            kv_slot->key_cache->ne[1] != config.text.kv_heads ||
            kv_slot->value_cache->ne[1] != config.text.kv_heads ||
            kv_slot->offset < 0 ||
            kv_slot->offset + tokens > kv_slot->key_cache->ne[2] ||
            kv_slot->value_cache->ne[2] != kv_slot->key_cache->ne[2]) {
            set_error(error, "invalid text decode KV cache slot");
            return false;
        }
        ggml_context *ctx = session.context();
        ggml_tensor *dst_k = ggml_view_3d(
            ctx, kv_slot->key_cache, config.text.head_dim, config.text.kv_heads,
            tokens, kv_slot->key_cache->nb[1], kv_slot->key_cache->nb[2],
            static_cast<size_t>(kv_slot->offset) * kv_slot->key_cache->nb[2]);
        ggml_tensor *dst_v = ggml_view_3d(
            ctx, kv_slot->value_cache, config.text.head_dim,
            config.text.kv_heads, tokens, kv_slot->value_cache->nb[1],
            kv_slot->value_cache->nb[2],
            static_cast<size_t>(kv_slot->offset) *
                kv_slot->value_cache->nb[2]);
        ggml_tensor *k_write = ggml_cpy(ctx, k_rope, dst_k);
        ggml_tensor *v_write = ggml_cpy(ctx, v_heads, dst_v);
        if (kv_output != nullptr)
            *kv_output = {k_write, v_write};
        if (read_kv_from_slot) {
            attention_key = ggml_view_3d(
                ctx, kv_slot->key_cache, config.text.head_dim,
                config.text.kv_heads, kv_slot->offset + tokens,
                kv_slot->key_cache->nb[1], kv_slot->key_cache->nb[2], 0);
            attention_value = ggml_view_3d(
                ctx, kv_slot->value_cache, config.text.head_dim,
                config.text.kv_heads, kv_slot->offset + tokens,
                kv_slot->value_cache->nb[1], kv_slot->value_cache->nb[2], 0);
        }
    } else if (past_kv != nullptr) {
        if (past_kv->key == nullptr || past_kv->value == nullptr ||
            past_kv->key->type != GGML_TYPE_F32 ||
            past_kv->value->type != GGML_TYPE_F32 ||
            past_kv->key->ne[0] != config.text.head_dim ||
            past_kv->value->ne[0] != config.text.head_dim ||
            past_kv->key->ne[1] != config.text.kv_heads ||
            past_kv->value->ne[1] != config.text.kv_heads ||
            past_kv->key->ne[2] != past_kv->value->ne[2]) {
            set_error(error, "invalid text decode KV input");
            return false;
        }
        attention_key =
            checked_concat(session.context(), past_kv->key, k_rope, 2);
        attention_value =
            checked_concat(session.context(), past_kv->value, v_heads, 2);
    }
    ggml_tensor *heads = nullptr;
    if (!build_text_causal_attention(session, q_rope, attention_key,
                                     attention_value, &heads, error))
        return false;
    // The attention head plane is heads * head_dim wide, which hidden_size
    // only matches for symmetric builds (2B); Qwen3-style blocks are wider
    // and o_proj projects the plane back down to hidden_size.
    ggml_tensor *heads_flat = ggml_reshape_2d(
        session.context(), heads,
        config.text.attention_heads * config.text.head_dim, tokens);
    ggml_tensor *attention = nullptr;
    if (!build_text_linear(session, heads_flat, name + "attn_output.weight",
                           config.text.hidden_size, &attention, error))
        return false;
    if (!checkpoint("attention", attention, "B,L,H"))
        return false;
    ggml_tensor *after_attention =
        checked_add(session.context(), input, attention);

    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *up = nullptr;
    if (!build_text_rms_norm(session, after_attention, name + "ffn_norm.weight",
                             &ffn_norm, error) ||
        !build_text_linear(session, ffn_norm, name + "ffn_gate.weight",
                           config.text.intermediate_size, &gate, error) ||
        !build_text_linear(session, ffn_norm, name + "ffn_up.weight",
                           config.text.intermediate_size, &up, error))
        return false;
    if (!checkpoint("ffn_norm", ffn_norm, "B,L,H") ||
        !checkpoint("ffn_gate", gate, "B,L,I") ||
        !checkpoint("ffn_up", up, "B,L,I"))
        return false;
    ggml_tensor *activated =
        ggml_swiglu_split(session.context(), gate, up);
    ggml_tensor *down = nullptr;
    if (!build_text_linear(session, activated, name + "ffn_down.weight",
                           config.text.hidden_size, &down, error))
        return false;
    if (!checkpoint("ffn_down", down, "B,L,H"))
        return false;
    *output = checked_add(session.context(), after_attention, down);
    char output_checkpoint[64];
    std::snprintf(output_checkpoint, sizeof(output_checkpoint),
                  "text.block.%02d.output", layer);
    return session.checkpoint({output_checkpoint, "text", "B,L,H"}, *output,
                              error);
}

} // namespace

bool build_text_decoder_block(GraphSession &session, int32_t layer,
                              ggml_tensor *input, const TextRotaryData &rotary,
                              const TextLayerKV *past_kv,
                              TextLayerKV *kv_output, ggml_tensor **output,
                              std::string *error) {
    return build_text_decoder_block_impl(session, layer, input, rotary, past_kv,
                                         nullptr, false, kv_output, output,
                                         error);
}

bool build_text_prefill(GraphSession &session, ggml_tensor *input,
                        const TextRotaryData &rotary,
                        const std::vector<int32_t> &image_positions,
                        const std::vector<ggml_tensor *> &deepstack_embeddings,
                        std::vector<TextLayerKV> *kv_outputs,
                        ggml_tensor **output, std::string *error,
                        const std::vector<TextLayerKVSlot> *kv_slots) {
    if (input == nullptr || output == nullptr ||
        input->ne[1] != rotary.sequence_length ||
        deepstack_embeddings.size() >
            session.model().config.vision.deepstack_layers.size() ||
        (kv_slots != nullptr &&
         kv_slots->size() != session.model().config.text.layer_count)) {
        set_error(error, "invalid text prefill input");
        return false;
    }
    if (std::any_of(image_positions.begin(), image_positions.end(),
                    [&input](int32_t position) {
                        return position < 0 || position >= input->ne[1];
                    }) ||
        !std::is_sorted(image_positions.begin(), image_positions.end()) ||
        std::adjacent_find(image_positions.begin(), image_positions.end()) !=
            image_positions.end()) {
        set_error(error, "invalid text deepstack positions");
        return false;
    }
    ggml_tensor *hidden = input;
    if (kv_outputs != nullptr) {
        kv_outputs->clear();
        kv_outputs->reserve(session.model().config.text.layer_count);
    }
    for (uint32_t layer = 0; layer < session.model().config.text.layer_count;
         ++layer) {
        TextLayerKV kv;
        if (!build_text_decoder_block_impl(
                session, static_cast<int32_t>(layer), hidden, rotary, nullptr,
                kv_slots == nullptr ? nullptr : &(*kv_slots)[layer], false,
                kv_outputs == nullptr ? nullptr : &kv, &hidden, error))
            return false;
        if (kv_outputs != nullptr) {
            kv_outputs->push_back(kv);
            char key_name[64];
            char value_name[64];
            std::snprintf(key_name, sizeof(key_name), "text.kv.%02u.key",
                          layer);
            std::snprintf(value_name, sizeof(value_name), "text.kv.%02u.value",
                          layer);
            if (!session.checkpoint(
                    {key_name, "text", "head_dim,kv_heads,sequence"}, kv.key,
                    error) ||
                !session.checkpoint(
                    {value_name, "text", "head_dim,kv_heads,sequence"},
                    kv.value, error))
                return false;
        }
        if (layer >= deepstack_embeddings.size())
            continue;
        ggml_tensor *features = deepstack_embeddings[layer];
        if (features == nullptr || features->type != GGML_TYPE_F32 ||
            features->ne[0] != hidden->ne[0] ||
            features->ne[1] != static_cast<int64_t>(image_positions.size())) {
            set_error(error, "invalid text deepstack embedding");
            return false;
        }
        // Qwen's _deepstack_process is additive and occurs immediately after
        // decoder layers 0, 1 and 2 at precisely the image-token positions.
        // Build it from slices instead of ggml_acc: Metal's ACC kernel does
        // not preserve the base outside the updated slice on all devices.
        ggml_tensor *updated = nullptr;
        int64_t cursor = 0;
        auto append = [&](ggml_tensor *piece) {
            updated = updated == nullptr ? piece
                                         : checked_concat(session.context(),
                                                          updated, piece, 1);
        };
        for (size_t first = 0; first < image_positions.size();) {
            size_t count = 1;
            while (first + count < image_positions.size() &&
                   image_positions[first + count] ==
                       image_positions[first] + static_cast<int32_t>(count))
                ++count;
            const int64_t position = image_positions[first];
            if (position > cursor) {
                append(
                    ggml_view_2d(session.context(), hidden, hidden->ne[0],
                                 position - cursor, hidden->nb[1],
                                 static_cast<size_t>(cursor) * hidden->nb[1]));
            }
            const size_t offset = static_cast<size_t>(position) * hidden->nb[1];
            ggml_tensor *destination = ggml_view_2d(
                session.context(), hidden, hidden->ne[0],
                static_cast<int64_t>(count), hidden->nb[1], offset);
            ggml_tensor *source =
                ggml_view_2d(session.context(), features, features->ne[0],
                             static_cast<int64_t>(count), features->nb[1],
                             first * features->nb[1]);
            append(checked_add(session.context(), destination, source));
            cursor = position + static_cast<int64_t>(count);
            first += count;
        }
        if (cursor < hidden->ne[1]) {
            append(ggml_view_2d(session.context(), hidden, hidden->ne[0],
                                hidden->ne[1] - cursor, hidden->nb[1],
                                static_cast<size_t>(cursor) * hidden->nb[1]));
        }
        hidden = updated;
    }
    *output = hidden;
    return session.checkpoint({"text.last_hidden", "text", "B,L,H"}, hidden,
                              error);
}

bool build_text_decode(GraphSession &session, ggml_tensor *input,
                       const TextRotaryData &rotary,
                       const std::vector<TextLayerKV> &past_kv,
                       std::vector<TextLayerKV> *new_kv, ggml_tensor **output,
                       std::string *error) {
    const auto &text = session.model().config.text;
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] != text.hidden_size ||
        input->ne[1] != rotary.sequence_length ||
        past_kv.size() != text.layer_count || new_kv == nullptr) {
        set_error(error, "invalid text decode input");
        return false;
    }
    new_kv->clear();
    new_kv->reserve(text.layer_count);
    ggml_tensor *hidden = input;
    for (uint32_t layer = 0; layer < text.layer_count; ++layer) {
        TextLayerKV current;
        if (!build_text_decoder_block(session, static_cast<int32_t>(layer),
                                      hidden, rotary, &past_kv[layer], &current,
                                      &hidden, error))
            return false;
        new_kv->push_back(current);
    }
    *output = hidden;
    return session.checkpoint({"text.decode_hidden", "text", "B,L,H"}, hidden,
                              error) &&
           session.checkpoint({"text.last_hidden", "text", "B,L,H"}, hidden,
                              error);
}

bool build_text_decode_inplace(GraphSession &session, ggml_tensor *input,
                               const TextRotaryData &rotary,
                               const std::vector<TextLayerKVSlot> &kv_slots,
                               std::vector<TextLayerKV> *kv_writes,
                               ggml_tensor **output, std::string *error) {
    const auto &text = session.model().config.text;
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] != text.hidden_size ||
        input->ne[1] != rotary.sequence_length ||
        kv_slots.size() != text.layer_count || kv_writes == nullptr) {
        set_error(error, "invalid text inplace decode input");
        return false;
    }
    kv_writes->clear();
    kv_writes->reserve(text.layer_count);
    ggml_tensor *hidden = input;
    for (uint32_t layer = 0; layer < text.layer_count; ++layer) {
        TextLayerKV write_ops;
        if (!build_text_decoder_block_impl(
                session, static_cast<int32_t>(layer), hidden, rotary, nullptr,
                &kv_slots[layer], true, &write_ops, &hidden, error))
            return false;
        kv_writes->push_back(write_ops);
    }
    *output = hidden;
    return session.checkpoint({"text.decode_hidden", "text", "B,L,H"}, hidden,
                              error) &&
           session.checkpoint({"text.last_hidden", "text", "B,L,H"}, hidden,
                              error);
}

bool build_text_next_logits(GraphSession &session, ggml_tensor *last_hidden,
                            ggml_tensor **output, std::string *error) {
    const auto &text = session.model().config.text;
    if (last_hidden == nullptr || output == nullptr ||
        last_hidden->type != GGML_TYPE_F32 ||
        last_hidden->ne[0] != text.hidden_size || last_hidden->ne[1] <= 0) {
        set_error(error, "invalid text next-logits input");
        return false;
    }
    const size_t offset =
        static_cast<size_t>(last_hidden->ne[1] - 1) * last_hidden->nb[1];
    ggml_tensor *token =
        ggml_view_2d(session.context(), last_hidden, last_hidden->ne[0], 1,
                     last_hidden->nb[1], offset);
    ggml_tensor *normalized = nullptr;
    if (!build_text_rms_norm(session, token, "wv.text.output_norm.weight",
                             &normalized, error) ||
        !build_text_linear(session, normalized, "wv.text.output.weight",
                           text.vocab_size, output, error))
        return false;
    return session.checkpoint({"text.next_logits", "text", "B,vocab"}, *output,
                              error);
}

bool build_text_linear(GraphSession &session, ggml_tensor *input,
                       const std::string &weight_name, int32_t output_size,
                       ggml_tensor **output, std::string *error) {
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        output_size <= 0) {
        set_error(error, "invalid text linear input");
        return false;
    }
    ggml_tensor *weight = session.weight(weight_name, error);
    if (weight == nullptr)
        return false;
    if (ggml_nelements(weight) !=
        input->ne[0] * static_cast<int64_t>(output_size)) {
        set_error(error, "invalid text linear weight shape: " + weight_name);
        return false;
    }
    ggml_tensor *matrix =
        ggml_reshape_2d(session.context(), weight, input->ne[0], output_size);
    // CPU BF16 reductions accumulate enough error across 28 decoder layers to
    // change generated tokens, so retain the FP32 promotion there. Metal has
    // a native BF16 matmul path; promoting every matrix inside each one-token
    // decode graph expands almost the entire model on every step and dominates
    // latency. Keep weights resident in their uploaded BF16 representation on
    // Metal instead of creating per-graph FP32 copies.
    if (matrix->type == GGML_TYPE_BF16 &&
        session.promotes_bf16_matmul_to_f32()) {
        matrix = ggml_cast(session.context(), matrix, GGML_TYPE_F32);
    }
    *output = checked_mul_mat(session.context(), matrix, input);
    return true;
}

bool build_text_input_embeddings(GraphSession &session,
                                 const std::vector<int32_t> &input_ids,
                                 const std::vector<int32_t> &image_positions,
                                 ggml_tensor *vision_embeddings,
                                 ggml_tensor **token_embeddings,
                                 ggml_tensor **output, std::string *error) {
    if (input_ids.empty() || token_embeddings == nullptr || output == nullptr) {
        set_error(error, "invalid text input embedding arguments");
        return false;
    }
    const Model &model = session.model();
    ggml_tensor *embedding_weight =
        session.weight("wv.text.token_embd.weight", error);
    if (embedding_weight == nullptr ||
        embedding_weight->ne[0] !=
            static_cast<int64_t>(model.config.text.hidden_size) ||
        embedding_weight->ne[1] !=
            static_cast<int64_t>(model.config.text.vocab_size)) {
        set_error(error, "invalid token embedding weight");
        return false;
    }
    if (std::any_of(input_ids.begin(), input_ids.end(), [&model](int32_t id) {
            return id < 0 ||
                   static_cast<uint32_t>(id) >= model.config.text.vocab_size;
        })) {
        set_error(error, "input token id outside vocabulary");
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *ids = ggml_new_tensor_1d(
        context, GGML_TYPE_I32, static_cast<int64_t>(input_ids.size()));
    if (!session.set_input(ids, input_ids.data(), ggml_nbytes(ids), error))
        return false;
    ggml_tensor *base =
        ggml_get_rows(context, embedding_weight, ids); // [hidden, sequence]
    *token_embeddings = base;
    if (!session.checkpoint(
            {"text.token_embeddings", "text", "hidden,sequence"}, base, error))
        return false;
    if (image_positions.empty()) {
        *output = base;
        return session.checkpoint(
            {"text.input_embeddings", "text", "hidden,sequence"}, *output,
            error);
    }
    if (vision_embeddings == nullptr ||
        vision_embeddings->type != GGML_TYPE_F32 ||
        vision_embeddings->ne[0] !=
            static_cast<int64_t>(model.config.text.hidden_size) ||
        vision_embeddings->ne[1] !=
            static_cast<int64_t>(image_positions.size())) {
        set_error(error, "vision embeddings do not match image-pad positions");
        return false;
    }
    if (std::any_of(image_positions.begin(), image_positions.end(),
                    [&input_ids](int32_t position) {
                        return position < 0 || static_cast<size_t>(position) >=
                                                   input_ids.size();
                    }) ||
        !std::is_sorted(image_positions.begin(), image_positions.end()) ||
        std::adjacent_find(image_positions.begin(), image_positions.end()) !=
            image_positions.end()) {
        set_error(error, "image-pad position outside sequence");
        return false;
    }
    // set_rows creates an alias view on some backends, while Metal ACC can
    // discard the base outside the updated slice. Rebuild the sequence from
    // unchanged text slices and replacement vision slices instead.
    ggml_tensor *value = nullptr;
    int64_t cursor = 0;
    auto append = [&](ggml_tensor *piece) {
        value =
            value == nullptr ? piece : checked_concat(context, value, piece, 1);
    };
    for (size_t first = 0; first < image_positions.size();) {
        size_t count = 1;
        while (first + count < image_positions.size() &&
               image_positions[first + count] ==
                   image_positions[first] + static_cast<int32_t>(count))
            ++count;
        const int64_t position = image_positions[first];
        if (position > cursor) {
            append(ggml_view_2d(context, base, base->ne[0], position - cursor,
                                base->nb[1],
                                static_cast<size_t>(cursor) * base->nb[1]));
        }
        ggml_tensor *source =
            ggml_view_2d(context, vision_embeddings, vision_embeddings->ne[0],
                         static_cast<int64_t>(count), vision_embeddings->nb[1],
                         first * vision_embeddings->nb[1]);
        append(source);
        cursor = position + static_cast<int64_t>(count);
        first += count;
    }
    if (cursor < base->ne[1]) {
        append(ggml_view_2d(context, base, base->ne[0], base->ne[1] - cursor,
                            base->nb[1],
                            static_cast<size_t>(cursor) * base->nb[1]));
    }
    *output = value;
    if (!session.checkpoint(
            {"text.input_embeddings", "text", "hidden,sequence"}, *output,
            error))
        return false;
    return true;
}

} // namespace wevision
