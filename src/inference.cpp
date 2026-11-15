#include "inference.hpp"

#include "graph_ops.hpp"
#include "kv_cache.hpp"
#include "mask.hpp"
#include "model_loader.hpp"
#include "profile.hpp"
#include "runtime.hpp"
#include "text.hpp"
#include "tokenizer.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace wevision {
namespace {

struct VisionHostOutput {
    int grid_h = 0;
    int grid_w = 0;
    int merged_tokens = 0;
    std::vector<float> merger;
    std::array<std::vector<float>, 3> deepstack;
    std::vector<std::vector<float>> mask_features;
};

// Maps graph profile stage tags to wevision_metrics fields that are assigned
// (not accumulated). Stages with aggregation rules, such as the transformer
// window or the logits_s1 detail window, are handled by the sample dispatch.
struct StageMetricEntry {
    MaskProfileTag tag;
    double wevision_metrics::*field;
};

constexpr StageMetricEntry kStageMetricTable[] = {
    {MaskProfileEncoder, &wevision_metrics::mask_encoder_ms},
    {MaskProfileFpn, &wevision_metrics::mask_fpn_ms},
    {MaskProfilePixelFusion, &wevision_metrics::mask_pixel_fusion_ms},
    {MaskProfileCoarse, &wevision_metrics::mask_coarse_ms},
    {MaskProfileAlphaS4, &wevision_metrics::mask_alpha_s4_ms},
    {MaskProfileAlphaS2, &wevision_metrics::mask_alpha_s2_ms},
};

// Assigns a sample to its metrics field; no-op for tags outside the table.
void assign_stage_metric(wevision_metrics *metrics, int tag,
                         double milliseconds) {
    for (const StageMetricEntry &entry : kStageMetricTable) {
        if (tag == static_cast<int>(entry.tag)) {
            metrics->*entry.field = milliseconds;
            return;
        }
    }
}

// All mask timing fields, in declaration order. Adding a metrics field only
// requires one new entry here; accumulation stays exhaustive.
constexpr double wevision_metrics::*kMaskMetricFields[] = {
    &wevision_metrics::mask_encoder_ms,
    &wevision_metrics::mask_transformer_ms,
    &wevision_metrics::mask_msda_ms,
    &wevision_metrics::mask_fpn_ms,
    &wevision_metrics::mask_pixel_fusion_ms,
    &wevision_metrics::mask_coarse_ms,
    &wevision_metrics::mask_alpha_s4_ms,
    &wevision_metrics::mask_alpha_s2_ms,
    &wevision_metrics::mask_alpha_s1_ms,
    &wevision_metrics::mask_postprocess_ms,
};

bool copy_tensor(GraphSession &session, ggml_tensor *tensor,
                 std::vector<float> *output, std::string *error) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        set_error(error, "cannot copy a non-f32 inference tensor");
        return false;
    }
    output->resize(static_cast<size_t>(ggml_nelements(tensor)));
    return session.copy_to_host(tensor, output->data(), ggml_nbytes(tensor),
                                error);
}

bool restore_premerge_row_major(std::vector<float> *feature, int grid_h,
                                int grid_w, int merge_size, size_t hidden_size,
                                std::string *error) {
    if (!feature || grid_h <= 0 || grid_w <= 0 || merge_size <= 0 ||
        grid_h % merge_size != 0 || grid_w % merge_size != 0 ||
        feature->size() != static_cast<size_t>(grid_h) * grid_w * hidden_size) {
        set_error(error, "invalid pre-merge vision feature layout");
        return false;
    }
    std::vector<float> restored(feature->size());
    const int block_width = grid_w / merge_size;
    for (int y = 0; y < grid_h; ++y) {
        const int block_y = y / merge_size;
        const int intra_y = y % merge_size;
        for (int x = 0; x < grid_w; ++x) {
            const int block_x = x / merge_size;
            const int intra_x = x % merge_size;
            const size_t source_token = static_cast<size_t>(
                (((block_y * block_width + block_x) * merge_size + intra_y) *
                     merge_size +
                 intra_x));
            const size_t target_token = static_cast<size_t>(y) * grid_w + x;
            std::copy_n(feature->data() + source_token * hidden_size,
                        hidden_size,
                        restored.data() + target_token * hidden_size);
        }
    }
    *feature = std::move(restored);
    return true;
}

bool run_vision(const Model &model, const BackendContext &backend,
                size_t graph_memory_bytes, const VisionImageInput &image,
                VisionHostOutput *output, std::string *error) {
    GraphSession session(model, backend, graph_memory_bytes);
    ggml_tensor *hidden = nullptr;
    if (!build_vision_patch_embed(session, image.patches, &hidden, error))
        return false;
    const int position_table_side =
        static_cast<int>(std::sqrt(static_cast<double>(
                             model.config.vision.position_table_size)) +
                         0.5);
    VisionPositionData positions;
    VisionRotaryData rotary;
    if (!build_vision_position_data(image.grid_thw[1], image.grid_thw[2],
                                    model.config.vision.spatial_merge_size,
                                    position_table_side, &positions)) {
        set_error(error, "cannot build vision position data");
        return false;
    }
    if (!build_vision_positioned_input(session, hidden, positions, &hidden,
                                       error) ||
        !build_vision_rotary_data(
            positions,
            model.config.vision.hidden_size / model.config.vision.head_count,
            static_cast<float>(model.config.vision.rope_theta), &rotary))
        return false;
    std::array<ggml_tensor *, 3> deep{};
    std::vector<ggml_tensor *> mask_features;
    for (uint32_t layer = 0; layer < model.config.vision.layer_count; ++layer) {
        if (!build_vision_block(session, static_cast<int>(layer), hidden,
                                rotary, &hidden, error))
            return false;
        if (std::find(model.config.mask.vision_feature_layers.begin(),
                      model.config.mask.vision_feature_layers.end(),
                      static_cast<int32_t>(layer)) !=
            model.config.mask.vision_feature_layers.end()) {
            mask_features.push_back(hidden);
        }
        const auto found =
            std::find(model.config.vision.deepstack_layers.begin(),
                      model.config.vision.deepstack_layers.end(),
                      static_cast<int32_t>(layer));
        if (found != model.config.vision.deepstack_layers.end()) {
            const size_t index = static_cast<size_t>(
                found - model.config.vision.deepstack_layers.begin());
            if (index >= deep.size() ||
                !build_vision_merger(session, hidden,
                                     "wv.vision.deepstack." +
                                         std::to_string(index),
                                     true, &deep[index], error))
                return false;
        }
    }
    ggml_tensor *merger = nullptr;
    if (!build_vision_merger(session, hidden, "wv.vision.merger", false,
                             &merger, error))
        return false;
    std::vector<ggml_tensor *> roots{merger};
    for (ggml_tensor *tensor : deep) {
        if (tensor)
            roots.push_back(tensor);
    }
    roots.insert(roots.end(), mask_features.begin(), mask_features.end());
    if (!session.compute(roots, error))
        return false;
    output->grid_h = image.grid_thw[1];
    output->grid_w = image.grid_thw[2];
    output->merged_tokens = static_cast<int>(merger->ne[1]);
    if (!copy_tensor(session, merger, &output->merger, error))
        return false;
    for (size_t index = 0; index < deep.size(); ++index) {
        if (!deep[index] || !copy_tensor(session, deep[index],
                                         &output->deepstack[index], error))
            return false;
    }
    output->mask_features.resize(mask_features.size());
    for (size_t index = 0; index < mask_features.size(); ++index) {
        if (!copy_tensor(session, mask_features[index],
                         &output->mask_features[index], error) ||
            !restore_premerge_row_major(&output->mask_features[index],
                                        output->grid_h, output->grid_w,
                                        model.config.vision.spatial_merge_size,
                                        model.config.vision.hidden_size, error))
            return false;
    }
    return true;
}

int32_t argmax(const std::vector<float> &logits, int32_t coordinate_begin,
               int32_t coordinate_end) {
    const int32_t best = static_cast<int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (best < coordinate_begin || best > coordinate_end)
        return best;
    // The MPS reference exposes FP16 logits. Reproduce its stable tie break
    // only for the contiguous <0>..<999> coordinate vocabulary, leaving text
    // and structural-tag selection on the original F32 logits.
    const ggml_fp16_t rounded = ggml_fp32_to_fp16(logits[best]);
    const float rounded_value = ggml_fp16_to_fp32(rounded);
    const float previous_value = ggml_fp16_to_fp32(
        rounded_value >= 0.0F ? static_cast<ggml_fp16_t>(rounded - 1)
                              : static_cast<ggml_fp16_t>(rounded + 1));
    const float tolerance = std::fabs(rounded_value - previous_value);
    for (int32_t index = coordinate_begin; index < best; ++index) {
        if (logits[best] - logits[index] <= tolerance)
            return index;
    }
    return best;
}

bool select_next_token(GraphSession &session, ggml_tensor *logits,
                       ggml_tensor *device_argmax, int32_t coordinate_begin,
                       int32_t coordinate_end, int32_t *next,
                       std::string *error) {
    if (!session.copy_to_host(device_argmax, next, sizeof(*next), error))
        return false;
    if (session.backend_kind() == WEVISION_BACKEND_METAL ||
        *next < coordinate_begin || *next > coordinate_end)
        return true;
    // Only coordinate tokens need the reference FP16 tie break. Read the
    // prefix ending at the winning coordinate instead of the full vocabulary.
    const size_t count = static_cast<size_t>(*next - coordinate_begin + 1);
    std::vector<float> coordinate_logits(count);
    if (!session.copy_to_host_range(logits, coordinate_logits.data(),
                                    static_cast<size_t>(coordinate_begin) *
                                        sizeof(float),
                                    count * sizeof(float), error))
        return false;
    *next = coordinate_begin +
            argmax(coordinate_logits, 0, static_cast<int32_t>(count) - 1);
    return true;
}

bool run_text(const Model &model, const BackendContext &backend,
              const QwenTokenizer &tokenizer, size_t graph_memory_bytes,
              uint32_t max_new_tokens, const VisionHostOutput &vision,
              const std::string &system_prompt, const std::string &prompt,
              std::string *text, std::vector<float> *object_hidden,
              wevision_metrics *metrics, std::string *error) {
    std::vector<int32_t> ids;
    const std::vector<ChatMessage> messages = {{"system", system_prompt, false},
                                               {"user", prompt, true}};
    if (!tokenizer.encode_chat(messages, true, vision.merged_tokens, &ids,
                               error))
        return false;
    const int32_t image_pad = tokenizer.token_id(kImagePadToken);
    const int32_t eos = tokenizer.token_id(kImEndToken);
    const int32_t coordinate_begin = tokenizer.token_id(kCoordinateBeginToken);
    const int32_t coordinate_end = tokenizer.token_id(kCoordinateEndToken);
    if (image_pad < 0 || eos < 0 || coordinate_begin < 0 ||
        coordinate_end < coordinate_begin) {
        set_error(error, "tokenizer lacks required multimodal tokens");
        return false;
    }
    std::vector<int32_t> image_positions;
    for (size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] == image_pad)
            image_positions.push_back(static_cast<int32_t>(index));
    }
    TextPositionIds positions;
    TextRotaryData rotary;
    if (!build_text_mrope_positions(
            ids, image_pad, 1, vision.grid_h, vision.grid_w,
            model.config.vision.spatial_merge_size, &positions, error) ||
        !build_text_rotary_data(
            positions, model.config.text.head_dim,
            static_cast<float>(model.config.text.rope_theta),
            model.config.text.mrope_sections, &rotary, error))
        return false;
    TextKVCache cache;
    if (!cache.initialize_backend(
            backend, model.config.text.layer_count,
            static_cast<uint32_t>(ids.size()) + max_new_tokens,
            model.config.text.kv_heads, model.config.text.head_dim, error))
        return false;
    int32_t next = -1;
    {
        ScopedTimer prefill_timer(&metrics->text_prefill_ms);
        GraphSession session(model, backend, graph_memory_bytes);
        ggml_tensor *vision_input = ggml_new_tensor_2d(
            session.context(), GGML_TYPE_F32, model.config.text.hidden_size,
            vision.merged_tokens);
        if (!session.set_input(vision_input, vision.merger.data(),
                               ggml_nbytes(vision_input), error))
            return false;
        ggml_tensor *ignored = nullptr;
        ggml_tensor *embedded = nullptr;
        if (!build_text_input_embeddings(session, ids, image_positions,
                                         vision_input, &ignored, &embedded,
                                         error))
            return false;
        std::vector<ggml_tensor *> deep_inputs;
        for (const auto &values : vision.deepstack) {
            ggml_tensor *tensor = ggml_new_tensor_2d(
                session.context(), GGML_TYPE_F32, model.config.text.hidden_size,
                vision.merged_tokens);
            if (!session.set_input(tensor, values.data(), ggml_nbytes(tensor),
                                   error))
                return false;
            deep_inputs.push_back(tensor);
        }
        std::vector<TextLayerKVSlot> slots;
        slots.reserve(cache.layer_count());
        for (uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            slots.push_back(
                {cache.key_tensor(layer), cache.value_tensor(layer), 0});
        }
        std::vector<TextLayerKV> graph_kv;
        ggml_tensor *hidden = nullptr;
        ggml_tensor *logits = nullptr;
        if (!build_text_prefill(session, embedded, rotary, image_positions,
                                deep_inputs, &graph_kv, &hidden, error,
                                &slots) ||
            !build_text_next_logits(session, hidden, &logits, error))
            return false;
        const bool device_argmax = backend.kind() != WEVISION_BACKEND_CPU;
        ggml_tensor *next_token =
            device_argmax ? ggml_argmax(session.context(), logits) : logits;
        std::vector<ggml_tensor *> roots;
        roots.reserve(graph_kv.size() * 2U + 1U);
        for (const TextLayerKV &kv : graph_kv) {
            roots.push_back(kv.key);
            roots.push_back(kv.value);
        }
        roots.push_back(next_token);
        if (!session.compute(roots, error))
            return false;
        if (device_argmax) {
            if (!select_next_token(session, logits, next_token,
                                   coordinate_begin, coordinate_end, &next,
                                   error))
                return false;
        } else {
            std::vector<float> host_logits(model.config.text.vocab_size);
            if (!session.copy_to_host(logits, host_logits.data(),
                                      host_logits.size() * sizeof(float),
                                      error))
                return false;
            next = argmax(host_logits, coordinate_begin, coordinate_end);
        }
        if (!cache.commit_inplace_append(static_cast<uint32_t>(ids.size()),
                                         error))
            return false;
    }
    std::vector<int32_t> generated;
    object_hidden->clear();
    ggml_gallocr_t decode_allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend.backend()));
    if (decode_allocator == nullptr) {
        set_error(error, "cannot create decode activation allocator");
        return false;
    }

    struct DecodeAllocatorGuard {
        ggml_gallocr_t allocator = nullptr;

        ~DecodeAllocatorGuard() {
            if (allocator != nullptr)
                ggml_gallocr_free(allocator);
        }
    } decode_allocator_guard{decode_allocator};

    ScopedTimer decode_timer(&metrics->text_decode_ms);
    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        generated.push_back(next);
        if (next == eos)
            break;
        GraphSession session(model, backend, graph_memory_bytes,
                             decode_allocator);
        ggml_tensor *ignored = nullptr;
        ggml_tensor *embedded = nullptr;
        if (!build_text_input_embeddings(session, {next}, {}, nullptr, &ignored,
                                         &embedded, error))
            return false;
        std::vector<TextLayerKVSlot> slots;
        slots.reserve(cache.layer_count());
        const int32_t cache_offset = static_cast<int32_t>(cache.length());
        for (uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            slots.push_back(
                {cache.key_tensor(layer), cache.value_tensor(layer),
                 cache_offset});
        }
        TextPositionIds current;
        current.sequence_length = 1;
        current.values.resize(3);
        for (int axis = 0; axis < 3; ++axis) {
            current.values[axis] =
                positions.values[static_cast<size_t>(axis) *
                                     positions.sequence_length +
                                 positions.sequence_length - 1] +
                step + 1;
        }
        TextRotaryData current_rotary;
        std::vector<TextLayerKV> kv_writes;
        ggml_tensor *hidden = nullptr;
        ggml_tensor *logits = nullptr;
        if (!build_text_rotary_data(
                current, model.config.text.head_dim,
                static_cast<float>(model.config.text.rope_theta),
                model.config.text.mrope_sections, &current_rotary, error) ||
            !build_text_decode_inplace(session, embedded, current_rotary, slots,
                                       &kv_writes, &hidden, error) ||
            !build_text_next_logits(session, hidden, &logits, error))
            return false;
        const bool device_argmax = backend.kind() != WEVISION_BACKEND_CPU;
        ggml_tensor *next_token =
            device_argmax ? ggml_argmax(session.context(), logits) : logits;
        std::vector<ggml_tensor *> roots;
        roots.reserve(kv_writes.size() * 2U + 2U);
        for (const TextLayerKV &kv : kv_writes) {
            roots.push_back(kv.key);
            roots.push_back(kv.value);
        }
        roots.push_back(next_token);
        roots.push_back(hidden);
        if (!session.compute(roots, error))
            return false;
        if (device_argmax) {
            if (!select_next_token(session, logits, next_token,
                                   coordinate_begin, coordinate_end, &next,
                                   error))
                return false;
        } else {
            std::vector<float> host_logits(model.config.text.vocab_size);
            if (!session.copy_to_host(logits, host_logits.data(),
                                      host_logits.size() * sizeof(float),
                                      error))
                return false;
            next = argmax(host_logits, coordinate_begin, coordinate_end);
        }
        if (next == static_cast<int32_t>(model.config.mask.seg_token_id)) {
            const size_t old_size = object_hidden->size();
            object_hidden->resize(old_size + model.config.text.hidden_size);
            if (!session.copy_to_host(
                    hidden, object_hidden->data() + old_size,
                    model.config.text.hidden_size * sizeof(float), error))
                return false;
        }
        if (!cache.commit_inplace_append(1, error))
            return false;
    }
    if (!tokenizer.decode(generated, text, error))
        return false;
    metrics->generated_tokens = static_cast<uint32_t>(generated.size());
    const size_t end = text->find(kImEndToken);
    if (end != std::string::npos)
        text->erase(end);
    return true;
}

struct SharedMaskInputs {
    ggml_context *context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *memory_tensor = nullptr;
    ggml_tensor *pixel_tensor = nullptr;
    MaskAlphaShared alpha;
    std::vector<int32_t> heights;
    std::vector<int32_t> widths;
    wevision_metrics metrics{};

    SharedMaskInputs() = default;
    ~SharedMaskInputs() {
        if (buffer != nullptr)
            ggml_backend_buffer_free(buffer);
        if (context != nullptr)
            ggml_free(context);
    }

    SharedMaskInputs(const SharedMaskInputs &) = delete;
    SharedMaskInputs &operator=(const SharedMaskInputs &) = delete;
};

bool build_shared_mask_inputs(const Model &model, const BackendContext &backend,
                              size_t graph_memory_bytes,
                              const VisionHostOutput &vision,
                              const MaskImageInput &image,
                              SharedMaskInputs *output, std::string *error) {
    if (!output ||
        vision.mask_features.size() != model.config.mask.msda_levels) {
        set_error(error, "invalid shared mask input");
        return false;
    }
    GraphSession session(model, backend, graph_memory_bytes);
    std::vector<ggml_tensor *> feature_inputs;
    for (size_t level = 0; level < vision.mask_features.size(); ++level) {
        ggml_tensor *raw = ggml_new_tensor_3d(session.context(), GGML_TYPE_F32,
                                              model.config.vision.hidden_size,
                                              vision.grid_w, vision.grid_h);
        if (!session.set_input(raw, vision.mask_features[level].data(),
                               ggml_nbytes(raw), error))
            return false;
        feature_inputs.push_back(raw);
    }
    ggml_tensor *raw_rgb = ggml_new_tensor_3d(session.context(), GGML_TYPE_F32,
                                              image.width, image.height, 3);
    if (!session.set_input(raw_rgb, image.rgb.data(), ggml_nbytes(raw_rgb),
                           error))
        return false;
    ggml_tensor *rgb =
        ggml_cont(session.context(),
                  ggml_permute(session.context(), raw_rgb, 1, 2, 0, 3));
    MaskProjectedFeatures projected;
    MaskEncoderInputs spatial;
    MaskFpnOutput fpn;
    ggml_tensor *memory = nullptr;
    ggml_tensor *pixel = nullptr;
    std::vector<GraphProfilePoint> points;
    if (!build_mask_encoder_projection(session, feature_inputs, &projected,
                                       error) ||
        !build_mask_encoder_inputs(session, projected, &spatial, error) ||
        !build_mask_spatial_encoder(
            session, spatial,
            static_cast<int32_t>(model.config.mask.encoder_layers), &memory,
            error))
        return false;
    points.push_back({"mask.encoder", memory, MaskProfileEncoder});
    if (!build_mask_fpn(session, memory, spatial, &fpn, error))
        return false;
    points.push_back({"mask.fpn", fpn.feature, MaskProfileFpn});
    if (!build_mask_pixel_fusion(session, fpn.feature, rgb, &pixel, error))
        return false;
    points.push_back({"mask.pixel_fusion", pixel, MaskProfilePixelFusion});
    MaskAlphaShared alpha_shared;
    if (!build_mask_alpha_shared(session, rgb, pixel, &alpha_shared, error))
        return false;

    const std::vector<ggml_tensor *> outputs = {
        memory,
        pixel,
        alpha_shared.semantic_s4,
        alpha_shared.semantic_s2,
        alpha_shared.semantic_s1,
        alpha_shared.rgb_s4,
        alpha_shared.rgb_s2,
        alpha_shared.rgb_s1,
    };
    std::vector<GraphProfileSample> samples;
    const DebugOptions debug = DebugOptions::from_environment();
    if (debug.stage_profile_disabled
            ? !session.compute(outputs, error)
            : !session.compute_profiled(outputs, points, &samples, error))
        return false;
    for (const GraphProfileSample &sample : samples)
        assign_stage_metric(&output->metrics, sample.tag, sample.milliseconds);
    output->context = ggml_init({ggml_tensor_overhead() * 12U, nullptr, true});
    if (output->context == nullptr) {
        set_error(error, "cannot create shared mask context");
        return false;
    }
    output->memory_tensor = ggml_new_tensor_2d(
        output->context, GGML_TYPE_F32, memory->ne[0], memory->ne[1]);
    output->pixel_tensor = ggml_new_tensor_3d(
        output->context, GGML_TYPE_F32, pixel->ne[0], pixel->ne[1],
        pixel->ne[2]);
    auto clone_3d = [&](ggml_tensor *src) {
        return ggml_new_tensor_3d(output->context, GGML_TYPE_F32, src->ne[0],
                                  src->ne[1], src->ne[2]);
    };
    output->alpha.semantic_s4 = clone_3d(alpha_shared.semantic_s4);
    output->alpha.semantic_s2 = clone_3d(alpha_shared.semantic_s2);
    output->alpha.semantic_s1 = clone_3d(alpha_shared.semantic_s1);
    output->alpha.rgb_s4 = clone_3d(alpha_shared.rgb_s4);
    output->alpha.rgb_s2 = clone_3d(alpha_shared.rgb_s2);
    output->alpha.rgb_s1 = clone_3d(alpha_shared.rgb_s1);
    output->buffer =
        ggml_backend_alloc_ctx_tensors(output->context, backend.backend());
    if (output->buffer == nullptr) {
        set_error(error, "cannot allocate shared mask backend buffer");
        return false;
    }
    const ggml_tensor *sources[8] = {
        memory,
        pixel,
        alpha_shared.semantic_s4,
        alpha_shared.semantic_s2,
        alpha_shared.semantic_s1,
        alpha_shared.rgb_s4,
        alpha_shared.rgb_s2,
        alpha_shared.rgb_s1,
    };
    ggml_tensor *destinations[8] = {
        output->memory_tensor,
        output->pixel_tensor,
        output->alpha.semantic_s4,
        output->alpha.semantic_s2,
        output->alpha.semantic_s1,
        output->alpha.rgb_s4,
        output->alpha.rgb_s2,
        output->alpha.rgb_s1,
    };
    ggml_backend_tensor_copy_n(sources, destinations, 8);
    output->heights = spatial.heights;
    output->widths = spatial.widths;
    return true;
}

struct MaskBatchInputs {
    const Model &model;
    const BackendContext &backend;
    size_t graph_memory_bytes;
    const VisionHostOutput &vision;
    const MaskImageInput &image;
    const ParsedResponse &parsed;
    const std::vector<float> &object_hidden;
    size_t object_offset;
    size_t object_count;
    const SharedMaskInputs *shared;
    float threshold;
    ggml_gallocr_t reusable_allocator = nullptr;
};

bool run_mask_batch(const MaskBatchInputs &batch, std::vector<float> *alpha,
                    std::vector<uint8_t> *mask, wevision_metrics *metrics,
                    std::string *error) {
    const size_t hidden_size = batch.model.config.text.hidden_size;
    const size_t total_objects = batch.object_hidden.size() / hidden_size;
    if (hidden_size == 0 || batch.object_hidden.size() % hidden_size != 0 ||
        batch.object_count == 0 ||
        batch.object_offset + batch.object_count > total_objects ||
        batch.parsed.boxes.size() < batch.object_offset + batch.object_count ||
        batch.vision.mask_features.size() !=
            batch.model.config.mask.msda_levels) {
        set_error(error, "mask response and object features do not match");
        return false;
    }
    GraphSession session(batch.model, batch.backend, batch.graph_memory_bytes,
                         batch.reusable_allocator);
    ggml_tensor *hidden = ggml_new_tensor_2d(session.context(), GGML_TYPE_F32,
                                             hidden_size, batch.object_count);
    if (!session.set_input(hidden,
                           batch.object_hidden.data() +
                               batch.object_offset * hidden_size,
                           ggml_nbytes(hidden), error))
        return false;
    std::vector<float> boxes(batch.object_count * 4U);
    for (size_t index = 0; index < batch.object_count; ++index) {
        const ParsedBox &box = batch.parsed.boxes[batch.object_offset + index];
        boxes[4 * index] = box.x1 / batch.image.original_width;
        boxes[4 * index + 1] = box.y1 / batch.image.original_height;
        boxes[4 * index + 2] = box.x2 / batch.image.original_width;
        boxes[4 * index + 3] = box.y2 / batch.image.original_height;
    }
    ggml_tensor *rgb = nullptr;
    MaskEncoderInputs spatial;
    MaskQueryOutput query;
    ggml_tensor *memory = nullptr;
    ggml_tensor *queries = nullptr;
    ggml_tensor *pixel = nullptr;
    ggml_tensor *coarse = nullptr;
    MaskAlphaOutput refined;
    std::vector<GraphProfilePoint> profile_points;
    const DebugOptions debug = DebugOptions::from_environment();
    const bool profile_mask_details = debug.mask_profile_details;
    if (batch.shared != nullptr) {
        memory = batch.shared->memory_tensor;
        pixel = batch.shared->pixel_tensor;
        spatial.heights = batch.shared->heights;
        spatial.widths = batch.shared->widths;
    } else {
        ggml_tensor *raw_rgb = ggml_new_tensor_3d(
            session.context(), GGML_TYPE_F32, batch.image.width,
            batch.image.height, 3);
        if (!session.set_input(raw_rgb, batch.image.rgb.data(),
                               ggml_nbytes(raw_rgb), error))
            return false;
        rgb = ggml_cont(session.context(),
                        ggml_permute(session.context(), raw_rgb, 1, 2, 0, 3));
        std::vector<ggml_tensor *> feature_inputs;
        for (size_t level = 0; level < batch.vision.mask_features.size();
             ++level) {
            ggml_tensor *raw =
                ggml_new_tensor_3d(session.context(), GGML_TYPE_F32,
                                   batch.model.config.vision.hidden_size,
                                   batch.vision.grid_w, batch.vision.grid_h);
            if (!session.set_input(raw,
                                   batch.vision.mask_features[level].data(),
                                   ggml_nbytes(raw), error))
                return false;
            feature_inputs.push_back(raw);
        }
        MaskProjectedFeatures projected;
        MaskFpnOutput fpn;
        if (!build_mask_encoder_projection(session, feature_inputs, &projected,
                                           error) ||
            !build_mask_encoder_inputs(session, projected, &spatial, error) ||
            !build_mask_spatial_encoder(
                session, spatial,
                static_cast<int32_t>(batch.model.config.mask.encoder_layers),
                &memory, error))
            return false;
        profile_points.push_back({"mask.encoder", memory, MaskProfileEncoder});
        if (!build_mask_fpn(session, memory, spatial, &fpn, error))
            return false;
        profile_points.push_back({"mask.fpn", fpn.feature, MaskProfileFpn});
        if (!build_mask_pixel_fusion(session, fpn.feature, rgb, &pixel, error))
            return false;
        profile_points.push_back(
            {"mask.pixel_fusion", pixel, MaskProfilePixelFusion});
    }
    if (!build_mask_query(session, hidden, boxes, &query, error))
        return false;
    if (!build_mask_transformer(session, query.content, query.position, boxes,
                                memory, spatial, &queries, error,
                                &profile_points))
        return false;
    profile_points.push_back(
        {"mask.transformer", queries, MaskProfileTransformer});
    if (!build_dynamic_mask_predictor(session, queries, pixel, &coarse, error))
        return false;
    profile_points.push_back({"mask.coarse", coarse, MaskProfileCoarse});
    if (!build_mask_alpha_cascade(
            session, rgb, pixel, queries, coarse, &refined, error,
            profile_mask_details ? &profile_points : nullptr,
            batch.shared != nullptr ? &batch.shared->alpha : nullptr))
        return false;
    profile_points.push_back(
        {"mask.stage_s4", refined.stage_s4, MaskProfileAlphaS4});
    profile_points.push_back(
        {"mask.stage_s2", refined.stage_s2, MaskProfileAlphaS2});
    profile_points.push_back(
        {"mask.logits_s1", refined.logits_s1, MaskProfileAlphaS1});
    std::vector<GraphProfileSample> profile_samples;
    if (debug.stage_profile_disabled
            ? !session.compute(refined.logits_s1, error)
            : !session.compute_profiled(refined.logits_s1, profile_points,
                                        &profile_samples, error))
        return false;
    // Stage tags make the dispatch name-independent. The transformer window
    // keeps the historical interval semantics: every sample from the encoder
    // checkpoint through the transformer checkpoint contributes to the
    // transformer total, and msda ends additionally contribute to the msda
    // subtotal. Alpha_s1 detail intervals aggregate into logits_s1.
    bool in_transformer = batch.shared != nullptr;
    for (const GraphProfileSample &sample : profile_samples) {
        switch (sample.tag) {
        case MaskProfileEncoder:
            assign_stage_metric(metrics, sample.tag, sample.milliseconds);
            in_transformer = true;
            continue;
        case MaskProfileTransformerWindow:
            if (in_transformer)
                metrics->mask_transformer_ms += sample.milliseconds;
            continue;
        case MaskProfileMsdaEnd:
            if (in_transformer)
                metrics->mask_transformer_ms += sample.milliseconds;
            metrics->mask_msda_ms += sample.milliseconds;
            continue;
        case MaskProfileTransformer:
            if (in_transformer)
                metrics->mask_transformer_ms += sample.milliseconds;
            in_transformer = false;
            continue;
        case MaskProfileAlphaS1Detail:
            if (profile_mask_details)
                std::fprintf(stderr, "mask_profile_ms %s=%.2f\n",
                             sample.name.c_str(), sample.milliseconds);
            metrics->mask_alpha_s1_ms += sample.milliseconds;
            continue;
        case MaskProfileAlphaS1:
            metrics->mask_alpha_s1_ms += sample.milliseconds;
            continue;
        default:
            // Direct stages: fpn, pixel_fusion, coarse, stage_s4, stage_s2.
            if (in_transformer)
                metrics->mask_transformer_ms += sample.milliseconds;
            assign_stage_metric(metrics, sample.tag, sample.milliseconds);
        }
    }
    std::vector<float> logits(
        static_cast<size_t>(ggml_nelements(refined.logits_s1)));
    if (!session.copy_to_host(refined.logits_s1, logits.data(),
                              logits.size() * sizeof(float), error))
        return false;
    ScopedTimer postprocess_timer(&metrics->mask_postprocess_ms);
    return postprocess_mask_logits(
        logits.data(), static_cast<int32_t>(batch.object_count),
        batch.image.width, batch.image.height, batch.image.original_width,
        batch.image.original_height, batch.threshold, alpha, mask, error);
}

void accumulate_mask_metrics(wevision_metrics *total,
                             const wevision_metrics &batch) {
    for (double wevision_metrics::*field : kMaskMetricFields)
        total->*field += batch.*field;
}

bool run_mask(const Model &model, const BackendContext &backend,
              size_t graph_memory_bytes, const VisionHostOutput &vision,
              const MaskImageInput &image, const ParsedResponse &parsed,
              const std::vector<float> &object_hidden, float threshold,
              std::vector<float> *alpha, std::vector<uint8_t> *mask,
              wevision_metrics *metrics, std::string *error) {
    const size_t hidden_size = model.config.text.hidden_size;
    if (hidden_size == 0 || object_hidden.size() % hidden_size != 0) {
        set_error(error, "invalid mask object features");
        return false;
    }
    const size_t object_count = object_hidden.size() / hidden_size;
    if (!backend.prefers_single_object_mask_batch() || object_count <= 1) {
        const MaskBatchInputs batch{model,         backend,  graph_memory_bytes,
                                    vision,        image,    parsed,
                                    object_hidden, 0,        object_count,
                                    nullptr,       threshold};
        return run_mask_batch(batch, alpha, mask, metrics, error);
    }

    SharedMaskInputs shared;
    if (!build_shared_mask_inputs(model, backend, graph_memory_bytes, vision,
                                  image, &shared, error))
        return false;
    accumulate_mask_metrics(metrics, shared.metrics);
    alpha->clear();
    mask->clear();
    const size_t pixels = static_cast<size_t>(image.original_width) *
                          static_cast<size_t>(image.original_height);
    alpha->reserve(object_count * pixels);
    mask->reserve(object_count * pixels);
    ggml_gallocr_t mask_allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend.backend()));
    if (mask_allocator == nullptr) {
        set_error(error, "cannot create mask activation allocator");
        return false;
    }
    struct MaskAllocatorGuard {
        ggml_gallocr_t allocator = nullptr;
        ~MaskAllocatorGuard() {
            if (allocator != nullptr)
                ggml_gallocr_free(allocator);
        }
    } mask_allocator_guard{mask_allocator};

    for (size_t offset = 0; offset < object_count; ++offset) {
        std::vector<float> batch_alpha;
        std::vector<uint8_t> batch_mask;
        wevision_metrics batch_metrics{};
        const MaskBatchInputs batch{model,          backend,  graph_memory_bytes,
                                    vision,         image,    parsed,
                                    object_hidden,  offset,   1,
                                    &shared,        threshold, mask_allocator};
        if (!run_mask_batch(batch, &batch_alpha, &batch_mask, &batch_metrics,
                            error))
            return false;
        alpha->insert(alpha->end(), batch_alpha.begin(), batch_alpha.end());
        mask->insert(mask->end(), batch_mask.begin(), batch_mask.end());
        accumulate_mask_metrics(metrics, batch_metrics);
    }
    return true;
}

} // namespace

static bool run_inference_impl(const Model &model,
                               const BackendContext &backend,
                               const QwenTokenizer &tokenizer,
                               size_t graph_memory_bytes,
                               uint32_t runtime_max_new_tokens,
                               const wevision_request &request,
                               InferenceOutput *output, std::string *error) {
    if (output == nullptr || request.image.rgb == nullptr ||
        request.image.width <= 0 || request.image.height <= 0 ||
        request.prompt == nullptr || request.task < WEVISION_TASK_DETECT ||
        request.task > WEVISION_TASK_MATTING) {
        set_error(error, "invalid inference request");
        return false;
    }
    const int32_t stride = request.image.row_stride == 0
                               ? request.image.width * 3
                               : request.image.row_stride;
    if (stride < request.image.width * 3) {
        set_error(error, "invalid RGB row stride");
        return false;
    }
    RgbImage source;
    source.width = request.image.width;
    source.height = request.image.height;
    source.pixels.resize(static_cast<size_t>(source.width) * source.height *
                         3U);
    for (int32_t y = 0; y < source.height; ++y) {
        std::memcpy(source.pixels.data() +
                        static_cast<size_t>(y) * source.width * 3U,
                    request.image.rgb + static_cast<size_t>(y) * stride,
                    static_cast<size_t>(source.width) * 3U);
    }
    VisionPreprocessConfig config;
    config.patch_size = model.config.vision.patch_size;
    config.temporal_patch_size = model.config.vision.temporal_patch_size;
    config.spatial_merge_size = model.config.vision.spatial_merge_size;
    if (request.vision_min_pixels != 0)
        config.min_pixels = static_cast<int>(request.vision_min_pixels);
    if (request.vision_max_pixels != 0)
        config.max_pixels = static_cast<int>(request.vision_max_pixels);
    InferenceOutput result;
    VisionImageInput vision_image;
    MaskImageInput mask_image;
    {
        ScopedTimer preprocess_timer(&result.metrics.preprocess_ms);
        if (!prepare_vision_image(source, config, &vision_image) ||
            ((request.task == WEVISION_TASK_SEGMENT ||
              request.task == WEVISION_TASK_MATTING) &&
             !prepare_mask_image(source, model.config.mask.cnn_image_size,
                                 &mask_image))) {
            set_error(error, "image preprocessing failed");
            return false;
        }
    }
    VisionHostOutput vision;
    {
        ScopedTimer vision_timer(&result.metrics.vision_ms);
        if (!run_vision(model, backend, graph_memory_bytes, vision_image,
                        &vision, error))
            return false;
    }
    const std::string system = request.system_prompt == nullptr
                                   ? "You are a helpful assistant."
                                   : request.system_prompt;
    const uint32_t max_tokens = request.max_new_tokens == 0
                                    ? runtime_max_new_tokens
                                    : request.max_new_tokens;
    std::vector<float> object_hidden;
    if (!run_text(model, backend, tokenizer, graph_memory_bytes, max_tokens,
                  vision, system, request.prompt, &result.text, &object_hidden,
                  &result.metrics, error) ||
        !parse_spatial_response(result.text, source.width, source.height,
                                &result.parsed, error))
        return false;
    result.image_width = source.width;
    result.image_height = source.height;
    if (request.task == WEVISION_TASK_SEGMENT ||
        request.task == WEVISION_TASK_MATTING) {
        const float threshold = std::isfinite(request.mask_threshold) &&
                                        request.mask_threshold >= 0.0F &&
                                        request.mask_threshold <= 1.0F
                                    ? request.mask_threshold
                                    : 0.5F;
        {
            ScopedTimer mask_timer(&result.metrics.mask_ms);
            if (!run_mask(model, backend, graph_memory_bytes, vision,
                          mask_image, result.parsed, object_hidden, threshold,
                          &result.alpha, &result.mask, &result.metrics, error))
                return false;
        }
    }
    *output = std::move(result);
    return true;
}

bool run_inference(const Model &model, const BackendContext &backend,
                   const QwenTokenizer &tokenizer, size_t graph_memory_bytes,
                   uint32_t runtime_max_new_tokens,
                   const wevision_request &request, InferenceOutput *output,
                   std::string *error) {
    try {
        return run_inference_impl(model, backend, tokenizer, graph_memory_bytes,
                                  runtime_max_new_tokens, request, output,
                                  error);
    } catch (const GraphShapeError &exception) {
        set_error(error, exception.what());
        return false;
    }
}

} // namespace wevision
