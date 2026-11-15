#include "mask.hpp"

#include "graph_ops.hpp"
#include "mask_ops.hpp"
#include "ms_deform_attn.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wevision {
using namespace mask_detail;

bool build_mask_query(GraphSession &session, ggml_tensor *obj_hidden,
                      const std::vector<float> &boxes, MaskQueryOutput *output,
                      std::string *error) {
    const int text_hidden =
        static_cast<int>(session.model().config.text.hidden_size);
    const int mask_hidden =
        static_cast<int>(session.model().config.mask.hidden_size);
    if (!obj_hidden || !output || obj_hidden->type != GGML_TYPE_F32 ||
        obj_hidden->ne[0] != text_hidden || obj_hidden->ne[1] <= 0 ||
        boxes.size() != static_cast<size_t>(obj_hidden->ne[1]) * 4U) {
        set_error(error, "invalid mask query input");
        return false;
    }
    const int n = static_cast<int>(obj_hidden->ne[1]);
    std::vector<float> encoded(static_cast<size_t>(n) * kBoxFourierDim);
    for (int i = 0; i < n; ++i) {
        const float *box = &boxes[static_cast<size_t>(i) * 4U];
        float values[4] = {
            box[0], box[1],
            0.2F * std::log(std::max(box[2] - box[0], 1e-6F)) + 0.5F,
            0.2F * std::log(std::max(box[3] - box[1], 1e-6F)) + 0.5F};
        for (int v = 0; v < 4; ++v)
            for (int f = 0; f < kBoxFourierFrequencies; ++f) {
                float p = 2.0F * static_cast<float>(M_PI) * values[v] *
                          std::ldexp(1.0F, f);
                encoded[static_cast<size_t>(i) * kBoxFourierDim +
                        v * 2 * kBoxFourierFrequencies + f] = std::sin(p);
                encoded[static_cast<size_t>(i) * kBoxFourierDim +
                        v * 2 * kBoxFourierFrequencies +
                        kBoxFourierFrequencies + f] = std::cos(p);
            }
    }
    ggml_context *c = session.context();
    ggml_tensor *fourier =
        ggml_new_tensor_2d(c, GGML_TYPE_F32, kBoxFourierDim, n);
    if (!session.set_input(fourier, encoded.data(), ggml_nbytes(fourier),
                           error))
        return false;
    ggml_tensor *content =
        linear(session, obj_hidden, "wv.mask.query_builder.content_proj",
               mask_hidden, error);
    if (!content)
        return false;
    content = layer_norm(session, content, "wv.mask.query_builder.content_norm",
                         error);
    ggml_tensor *position =
        linear(session, fourier, "wv.mask.query_builder.box_position.mlp.0",
               mask_hidden, error);
    if (!position)
        return false;
    position = ggml_gelu_erf(c, position);
    position =
        linear(session, position, "wv.mask.query_builder.box_position.mlp.2",
               mask_hidden, error);
    position = layer_norm(session, position,
                          "wv.mask.query_builder.position_norm", error);
    if (!content || !position)
        return false;
    output->content = content;
    output->position = position;
    return session.checkpoint({"mask.query.content", "mask", "objects,H"},
                              content, error) &&
           session.checkpoint({"mask.query.position", "mask", "objects,H"},
                              position, error);
}

bool build_mask_encoder_projection(GraphSession &session,
                                   const std::vector<ggml_tensor *> &features,
                                   MaskProjectedFeatures *output,
                                   std::string *error) {
    const int64_t vision_hidden =
        static_cast<int64_t>(session.model().config.vision.hidden_size);
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    if (!output || features.size() != session.model().config.mask.msda_levels) {
        set_error(error, "invalid mask encoder levels");
        return false;
    }
    output->maps.clear();
    ggml_context *c = session.context();
    for (size_t level = 0; level < features.size(); ++level) {
        ggml_tensor *x = features[level];
        if (!x || x->type != GGML_TYPE_F32 || x->ne[0] != vision_hidden ||
            x->ne[1] <= 0 || x->ne[2] <= 0) {
            set_error(error, "invalid mask encoder feature");
            return false;
        }
        const int64_t pixels = x->ne[1] * x->ne[2];
        std::string p =
            "wv.mask.visual_encoder.input_proj." + std::to_string(level);
        ggml_tensor *w = session.weight(p + ".0.weight", error);
        ggml_tensor *scale = session.weight(p + ".1.weight", error);
        ggml_tensor *bias = session.weight(p + ".1.bias", error);
        if (!w || !scale || !bias)
            return false;
        if (ggml_nelements(w) != mask_hidden * vision_hidden ||
            ggml_nelements(scale) != mask_hidden ||
            ggml_nelements(bias) != mask_hidden) {
            set_error(error,
                      "invalid mask encoder projection weight shape: " + p);
            return false;
        }
        ggml_tensor *flat = ggml_reshape_2d(c, x, vision_hidden, pixels);
        ggml_tensor *y = checked_mul_mat(
            c, ggml_reshape_2d(c, w, vision_hidden, mask_hidden), flat);
        y = ggml_reshape_3d(c, y, mask_hidden, x->ne[1], x->ne[2]);
        // ggml_group_norm defines channels as dimension 2.  Adapt the
        // internal C,W,H layout at this boundary and return to it afterwards.
        ggml_tensor *nhw = ggml_cont(c, ggml_permute(c, y, 2, 0, 1, 3));
        nhw = ggml_group_norm(c, nhw, mask_norm_groups(session),
                              session.model().config.mask.norm_eps);
        ggml_tensor *scale_tensor =
            ggml_reshape_4d(c, scale, 1, 1, mask_hidden, 1);
        ggml_tensor *b = ggml_reshape_4d(c, bias, 1, 1, mask_hidden, 1);
        y = ggml_cont(
            c, ggml_permute(
                   c, checked_add(c, checked_mul(c, nhw, scale_tensor), b), 1,
                   2, 0, 3));
        output->maps.push_back(y);
        // ggml stores the feature map as C,W,H.  The reference hook writes a
        // contiguous PyTorch B,C,H,W tensor, so materialize a W,H,C view before
        // tracing; FileTraceSink reverses ggml dimensions on export. The copy
        // is trace-only and skipped entirely while no sink is bound.
        if (session.trace_attached()) {
            std::string n =
                "mask.visual_encoder.projected." + std::to_string(level);
            ggml_tensor *traced = ggml_cont(c, ggml_permute(c, y, 2, 0, 1, 3));
            traced =
                ggml_reshape_4d(c, traced, x->ne[1], x->ne[2], mask_hidden, 1);
            if (!session.checkpoint({n, "mask", "B,C,H,W"}, traced, error))
                return false;
        }
    }
    return true;
}

bool build_mask_encoder_inputs(GraphSession &session,
                               const MaskProjectedFeatures &projected,
                               MaskEncoderInputs *output, std::string *error) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    if (!output ||
        projected.maps.size() != session.model().config.mask.msda_levels) {
        set_error(error, "invalid mask projected features");
        return false;
    }
    ggml_context *c = session.context();
    ggml_tensor *level_embed =
        session.weight("wv.mask.visual_encoder.level_embed", error);
    if (!level_embed || level_embed->type != GGML_TYPE_F32 ||
        level_embed->ne[0] != mask_hidden ||
        level_embed->ne[1] !=
            static_cast<int64_t>(session.model().config.mask.msda_levels)) {
        set_error(error, "invalid mask level embedding");
        return false;
    }
    output->src = nullptr;
    output->pos = nullptr;
    output->heights.clear();
    output->widths.clear();
    for (size_t level = 0; level < projected.maps.size(); ++level) {
        ggml_tensor *map = projected.maps[level];
        if (!map || map->type != GGML_TYPE_F32 || map->ne[0] != mask_hidden ||
            !ggml_is_contiguous(map)) {
            set_error(error, "invalid projected map layout");
            return false;
        }
        const int32_t width = static_cast<int32_t>(map->ne[1]);
        const int32_t height = static_cast<int32_t>(map->ne[2]);
        const int32_t tokens = width * height;
        ggml_tensor *flat = ggml_reshape_2d(c, map, mask_hidden, tokens);
        ggml_tensor *level_vector = ggml_view_1d(
            c, level_embed, mask_hidden,
            static_cast<int64_t>(level) * mask_hidden * sizeof(float));
        std::vector<float> position(static_cast<size_t>(mask_hidden) * tokens);
        for (int32_t y = 0; y < height; ++y)
            for (int32_t x = 0; x < width; ++x) {
                const float py = (static_cast<float>(y) + 0.5F) / height *
                                 (2.0F * static_cast<float>(M_PI));
                const float px = (static_cast<float>(x) + 0.5F) / width *
                                 (2.0F * static_cast<float>(M_PI));
                const int32_t token = x + width * y;
                for (int32_t index = 0; index < mask_hidden / 2; ++index) {
                    const float divisor = std::pow(
                        10000.0F, 2.0F * (index / 2) /
                                      static_cast<float>(mask_hidden / 2));
                    position[static_cast<size_t>(index) +
                             static_cast<size_t>(mask_hidden) * token] =
                        index % 2 == 0 ? std::sin(py / divisor)
                                       : std::cos(py / divisor);
                    position[static_cast<size_t>(mask_hidden / 2 + index) +
                             static_cast<size_t>(mask_hidden) * token] =
                        index % 2 == 0 ? std::sin(px / divisor)
                                       : std::cos(px / divisor);
                }
            }
        ggml_tensor *pos =
            ggml_new_tensor_2d(c, GGML_TYPE_F32, mask_hidden, tokens);
        if (!session.set_input(pos, position.data(), ggml_nbytes(pos), error))
            return false;
        pos = checked_add(c, pos,
                          ggml_reshape_2d(c, level_vector, mask_hidden, 1));
        output->src =
            output->src ? checked_concat(c, output->src, flat, 1) : flat;
        output->pos =
            output->pos ? checked_concat(c, output->pos, pos, 1) : pos;
        output->heights.push_back(height);
        output->widths.push_back(width);
    }
    return session.checkpoint({"mask.visual_encoder.src", "mask", "B,tokens,H"},
                              output->src, error) &&
           session.checkpoint({"mask.visual_encoder.pos", "mask", "B,tokens,H"},
                              output->pos, error);
}

bool build_mask_spatial_encoder(GraphSession &session,
                                const MaskEncoderInputs &input,
                                int32_t layer_count, ggml_tensor **output,
                                std::string *error) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    if (!output || !input.src || !input.pos || layer_count <= 0 ||
        input.src->type != GGML_TYPE_F32 || input.pos->type != GGML_TYPE_F32 ||
        input.src->ne[0] != mask_hidden ||
        !ggml_are_same_shape(input.src, input.pos) || input.heights.empty() ||
        input.heights.size() != input.widths.size()) {
        set_error(error, "invalid mask spatial encoder input");
        return false;
    }
    ggml_context *c = session.context();
    ggml_tensor *src = input.src;
    const int64_t total_tokens = src->ne[1];
    int64_t expected_tokens = 0;
    for (size_t level = 0; level < input.heights.size(); ++level) {
        if (input.heights[level] <= 0 || input.widths[level] <= 0) {
            set_error(error, "invalid mask spatial shape");
            return false;
        }
        expected_tokens +=
            static_cast<int64_t>(input.heights[level]) * input.widths[level];
    }
    if (expected_tokens != total_tokens) {
        set_error(error, "mask spatial shapes do not match token count");
        return false;
    }
    for (int32_t layer = 0; layer < layer_count; ++layer) {
        const std::string prefix = "wv.mask.visual_encoder.encoder.layers." +
                                   std::to_string(layer) + ".";
        ggml_tensor *depthwise =
            session.weight(prefix + "depthwise.weight", error);
        ggml_tensor *pointwise =
            session.weight(prefix + "pointwise.weight", error);
        if (!depthwise || !pointwise)
            return false;
        if (depthwise->type != GGML_TYPE_F32 ||
            pointwise->type != GGML_TYPE_F32 || depthwise->ne[0] != 3 ||
            depthwise->ne[1] != 3 || depthwise->ne[2] != 1 ||
            depthwise->ne[3] != mask_hidden ||
            ggml_nelements(pointwise) != mask_hidden * mask_hidden) {
            set_error(error,
                      "invalid spatial encoder convolution weight shape: " +
                          prefix);
            return false;
        }
        ggml_tensor *mixed = nullptr;
        int64_t offset = 0;
        ggml_tensor *src_with_pos = checked_add(c, src, input.pos);
        for (size_t level = 0; level < input.heights.size(); ++level) {
            const int64_t width = input.widths[level];
            const int64_t height = input.heights[level];
            const int64_t tokens = width * height;
            ggml_tensor *tokens_view =
                ggml_view_2d(c, src_with_pos, mask_hidden,
                             tokens, src_with_pos->nb[1],
                             offset * src_with_pos->nb[1]);
            // Conv2D consumes W,H,C,N. Materialize this standard layout so
            // ggml uses its regular direct-depthwise path (the CWH fast path
            // has an incompatible kernel-stride contract for GGUF weights).
            ggml_tensor *feature =
                ggml_reshape_3d(c, tokens_view, mask_hidden, width, height);
            ggml_tensor *whc =
                ggml_cont(c, ggml_permute(c, feature, 2, 0, 1, 3));
            ggml_tensor *convolved =
                checked_conv_2d_dw_direct(c, depthwise, whc, 1, 1, 1, 1, 1, 1);
            convolved = ggml_cont(c, convolved);
            convolved = ggml_gelu_erf(c, convolved);
            ggml_tensor *cwh = ggml_permute(c, convolved, 1, 2, 0, 3);
            cwh = ggml_cont(c, cwh);
            ggml_tensor *flat = ggml_reshape_2d(c, cwh, mask_hidden, tokens);
            ggml_tensor *pointwise_matrix =
                ggml_reshape_2d(c, pointwise, mask_hidden, mask_hidden);
            flat = checked_mul_mat(c, pointwise_matrix, flat);
            mixed = mixed ? checked_concat(c, mixed, flat, 1) : flat;
            offset += tokens;
        }
        ggml_tensor *normalized = layer_norm(
            session, checked_add(c, src, mixed), prefix + "norm1", error);
        if (!normalized)
            return false;
        ggml_tensor *ffn = linear(session, normalized, prefix + "linear1",
                                  static_cast<int>(4 * mask_hidden), error);
        if (!ffn)
            return false;
        ffn = ggml_gelu_erf(c, ffn);
        ffn = linear(session, ffn, prefix + "linear2",
                     static_cast<int>(mask_hidden), error);
        if (!ffn)
            return false;
        src = layer_norm(session, checked_add(c, normalized, ffn),
                         prefix + "norm2", error);
        if (!src)
            return false;
    }
    *output = src;
    return session.checkpoint(
        {"mask.visual_encoder.memory", "mask", "B,tokens,H"}, src, error);
}

bool build_mask_transformer(GraphSession &session, ggml_tensor *content,
                            ggml_tensor *position,
                            const std::vector<float> &boxes,
                            ggml_tensor *memory,
                            const MaskEncoderInputs &spatial,
                            ggml_tensor **output, std::string *error,
                            std::vector<GraphProfilePoint> *profile_points) {
    const auto &config = session.model().config.mask;
    const int64_t mask_hidden = static_cast<int64_t>(config.hidden_size);
    if (!output || !content || !position || !memory ||
        content->type != GGML_TYPE_F32 || position->type != GGML_TYPE_F32 ||
        memory->type != GGML_TYPE_F32 || content->ne[0] != mask_hidden ||
        !ggml_are_same_shape(content, position) ||
        memory->ne[0] != mask_hidden ||
        boxes.size() != static_cast<size_t>(content->ne[1]) * 4U ||
        config.msda_levels != spatial.heights.size() ||
        spatial.widths.size() != spatial.heights.size()) {
        set_error(error, "invalid mask transformer input");
        return false;
    }
    const int64_t queries = content->ne[1];
    const int64_t tokens = memory->ne[1];
    const int32_t heads = static_cast<int32_t>(config.msda_heads),
                  head_dim = static_cast<int32_t>(config.hidden_size /
                                                  config.msda_heads),
                  levels = static_cast<int32_t>(spatial.heights.size()),
                  points = static_cast<int32_t>(config.msda_points);
    const int32_t per_head = levels * points;
    ggml_context *c = session.context();
    ggml_tensor *target = content;
    std::vector<float> reference(static_cast<size_t>(2) * per_head * heads *
                                 queries);
    std::vector<float> scale(reference.size());
    for (int64_t query = 0; query < queries; ++query) {
        const float x1 = std::max(0.0F, std::min(1.0F, boxes[4 * query]));
        const float y1 = std::max(0.0F, std::min(1.0F, boxes[4 * query + 1]));
        const float x2 = std::max(0.0F, std::min(1.0F, boxes[4 * query + 2]));
        const float y2 = std::max(0.0F, std::min(1.0F, boxes[4 * query + 3]));
        const float cx = (x1 + x2) * 0.5F, cy = (y1 + y2) * 0.5F;
        const float width = std::max(x2 - x1, 1e-4F),
                    height = std::max(y2 - y1, 1e-4F);
        for (int32_t head = 0; head < heads; ++head)
            for (int32_t item = 0; item < per_head; ++item) {
                const size_t base = 2U * (static_cast<size_t>(item) +
                                          static_cast<size_t>(per_head) *
                                              (head + heads * query));
                reference[base] = cx;
                reference[base + 1] = cy;
                scale[base] = width * 0.5F / points;
                scale[base + 1] = height * 0.5F / points;
            }
    }
    ggml_tensor *base =
        ggml_new_tensor_4d(c, GGML_TYPE_F32, 2, per_head, heads, queries);
    ggml_tensor *offset_scale =
        ggml_new_tensor_4d(c, GGML_TYPE_F32, 2, per_head, heads, queries);
    if (!session.set_input(base, reference.data(), ggml_nbytes(base), error) ||
        !session.set_input(offset_scale, scale.data(),
                           ggml_nbytes(offset_scale), error))
        return false;
    for (uint32_t layer = 0; layer < config.transformer_layers; ++layer) {
        const std::string prefix =
            "wv.mask.decoder.blk." + std::to_string(layer) + ".";
        ggml_tensor *qk = checked_add(c, target, position);
        ggml_tensor *packed =
            linear_parameters(session, qk, prefix + "self_attn.in_proj_weight",
                              prefix + "self_attn.in_proj_bias",
                              static_cast<int>(3 * mask_hidden), error);
        if (!packed)
            return false;
        ggml_tensor *in_weight =
            session.weight(prefix + "self_attn.in_proj_weight", error);
        ggml_tensor *in_bias =
            session.weight(prefix + "self_attn.in_proj_bias", error);
        if (!in_weight || !in_bias)
            return false;
        ggml_tensor *value_weight =
            ggml_view_2d(c, in_weight, mask_hidden, mask_hidden,
                         in_weight->nb[1], 2 * mask_hidden * in_weight->nb[1]);
        ggml_tensor *value_bias = ggml_view_1d(c, in_bias, mask_hidden,
                                               2 * mask_hidden * sizeof(float));
        ggml_tensor *value_linear = checked_mul_mat(c, value_weight, target);
        value_linear = checked_add(
            c, value_linear, ggml_reshape_2d(c, value_bias, mask_hidden, 1));
        auto self_slice = [&](int64_t offset) {
            ggml_tensor *slice =
                ggml_view_2d(c, packed, mask_hidden, queries, packed->nb[1],
                             offset * sizeof(float));
            return ggml_reshape_3d(c, ggml_cont(c, slice), head_dim, heads,
                                   queries);
        };
        ggml_tensor *q = self_slice(0);
        ggml_tensor *k = self_slice(mask_hidden);
        ggml_tensor *v =
            ggml_reshape_3d(c, value_linear, head_dim, heads, queries);
        if (layer == 0 &&
            !session.checkpoint(
                {"mask.transformer.layer.00.self_value", "mask", "B,objects,H"},
                ggml_reshape_2d(c, v, mask_hidden, queries), error))
            return false;
        q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
        // Object self-attention is small and unmasked. Express it with standard
        // graph ops because Metal flash-attention does not support this shape.
        ggml_tensor *scores = checked_mul_mat(c, k, q);
        scores = ggml_scale(c, scores,
                            1.0F / std::sqrt(static_cast<float>(head_dim)));
        scores = ggml_soft_max(c, scores);
        ggml_tensor *value_transposed =
            ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));
        ggml_tensor *attended = checked_mul_mat(c, value_transposed, scores);
        attended = ggml_cont(c, ggml_permute(c, attended, 0, 2, 1, 3));
        attended = ggml_reshape_2d(c, attended, mask_hidden, queries);
        ggml_tensor *self_out = linear_parameters(
            session, attended, prefix + "self_attn.out_proj.weight",
            prefix + "self_attn.out_proj.bias", static_cast<int>(mask_hidden),
            error);
        if (!self_out)
            return false;
        if (layer == 0 &&
            !session.checkpoint(
                {"mask.transformer.layer.00.self_out", "mask", "B,objects,H"},
                self_out, error))
            return false;
        target = layer_norm(session, checked_add(c, target, self_out),
                            prefix + "norm1", error);
        if (!target)
            return false;
        if (layer == 0 &&
            !session.checkpoint(
                {"mask.transformer.layer.00.norm1", "mask", "B,objects,H"},
                target, error))
            return false;

        ggml_tensor *cross_query = checked_add(c, target, position);
        ggml_tensor *value =
            linear(session, memory, prefix + "cross_attn.value_proj",
                   static_cast<int>(mask_hidden), error);
        ggml_tensor *offsets =
            linear(session, cross_query, prefix + "cross_attn.sampling_offsets",
                   2 * per_head * heads, error);
        ggml_tensor *weights = linear(session, cross_query,
                                      prefix + "cross_attn.attention_weights",
                                      per_head * heads, error);
        if (!value || !offsets || !weights)
            return false;
        value = ggml_reshape_3d(c, value, head_dim, heads, tokens);
        offsets = ggml_reshape_4d(c, offsets, 2, per_head, heads, queries);
        weights = ggml_soft_max(
            c, ggml_reshape_3d(c, weights, per_head, heads, queries));
        ggml_tensor *locations =
            checked_add(c, base, checked_mul(c, offsets, offset_scale));
        profile_point(profile_points, MaskProfileTransformerWindow,
                      "mask.msda." + std::to_string(layer) + ".start",
                      locations);
        MSDeformAttnConfig sampler;
        sampler.head_dim = head_dim;
        sampler.heads = heads;
        sampler.levels = levels;
        sampler.points = points;
        sampler.heights = spatial.heights;
        sampler.widths = spatial.widths;
        ggml_tensor *sampled = nullptr;
        if (!build_ms_deform_attn_sampler(session, cross_query, value,
                                          locations, weights, sampler, &sampled,
                                          error))
            return false;
        profile_point(profile_points, MaskProfileMsdaEnd,
                      "mask.msda." + std::to_string(layer) + ".end", sampled);
        ggml_tensor *cross_out =
            linear(session, sampled, prefix + "cross_attn.output_proj",
                   static_cast<int>(mask_hidden), error);
        if (!cross_out)
            return false;
        if (layer == 0 &&
            !session.checkpoint(
                {"mask.transformer.layer.00.cross_out", "mask", "B,objects,H"},
                cross_out, error))
            return false;
        target = layer_norm(session, checked_add(c, target, cross_out),
                            prefix + "norm2", error);
        if (!target)
            return false;
        if (layer == 0 &&
            !session.checkpoint(
                {"mask.transformer.layer.00.norm2", "mask", "B,objects,H"},
                target, error))
            return false;
        ggml_tensor *ffn = linear(session, target, prefix + "linear1",
                                  static_cast<int>(4 * mask_hidden), error);
        if (!ffn)
            return false;
        ffn = linear(session, ggml_gelu_erf(c, ffn), prefix + "linear2",
                     static_cast<int>(mask_hidden), error);
        if (!ffn)
            return false;
        target = layer_norm(session, checked_add(c, target, ffn),
                            prefix + "norm3", error);
        if (!target)
            return false;
        const std::string checkpoint = std::string("mask.transformer.layer.") +
                                       (layer < 10 ? "0" : "") +
                                       std::to_string(layer) + ".output";
        if (!session.checkpoint({checkpoint, "mask", "B,objects,H"}, target,
                                error))
            return false;
    }
    *output = target;
    return session.checkpoint(
        {"mask.transformer.output", "mask", "B,objects,H"}, target, error);
}

} // namespace wevision
