#include "mask.hpp"

#include "graph_ops.hpp"
#include "mask_ops.hpp"
#include "pixel_ops.hpp"
#include "profile.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wevision {
using namespace mask_detail;

namespace {

ggml_tensor *pixel_shuffle_stage_whc(GraphSession &session, ggml_tensor *x,
                                     const std::string &prefix,
                                     std::string *error) {
    ggml_tensor *y = conv2d_whc(session, x, prefix + ".conv.weight", 1, error);
    if (!y)
        return nullptr;
    y = add_channel_bias_whc(session, y, prefix + ".conv.bias", error);
    if (!y)
        return nullptr;
    y = pixel_shuffle2x_whc(session, y, error);
    if (!y)
        return nullptr;
    return ggml_gelu_erf(session.context(), y);
}

ggml_tensor *
alpha_project_whc(GraphSession &session, ggml_tensor *x,
                  const std::string &prefix, std::string *error,
                  std::vector<GraphProfilePoint> *profile = nullptr,
                  const std::string &profile_prefix = {}) {
    ggml_tensor *y = conv2d_whc(session, x, prefix + ".0.weight", 1, error);
    if (!y)
        return nullptr;
    profile_point(profile, MaskProfileAlphaS1Detail, profile_prefix + ".conv",
                  y);
    y = group_norm_affine_whc(session, y, prefix + ".1",
                              mask_norm_groups(session), error);
    if (!y)
        return nullptr;
    profile_point(profile, MaskProfileAlphaS1Detail, profile_prefix + ".norm",
                  y);
    y = ggml_gelu_erf(session.context(), y);
    profile_point(profile, MaskProfileAlphaS1Detail, profile_prefix + ".gelu",
                  y);
    return y;
}

ggml_tensor *alpha_condition_whc(GraphSession &session, ggml_tensor *x,
                                 ggml_tensor *queries, const std::string &name,
                                 std::string *error) {
    ggml_context *c = session.context();
    const int64_t channels = x->ne[2], batch = x->ne[3];
    ggml_tensor *params =
        linear(session, queries, name, static_cast<int>(channels * 2), error);
    if (!params || params->ne[1] != batch)
        return nullptr;
    ggml_tensor *scale =
        ggml_view_2d(c, params, channels, batch, params->nb[1], 0);
    ggml_tensor *bias = ggml_view_2d(c, params, channels, batch, params->nb[1],
                                     channels * sizeof(float));
    scale = ggml_scale(c, ggml_tanh(c, scale), 0.1F);
    bias = ggml_scale(c, bias, 0.1F);
    ggml_tensor *one = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
    const float one_value = 1.0F;
    if (!session.set_input(one, &one_value, sizeof(one_value), error))
        return nullptr;
    scale = checked_add(c, scale, one);
    scale = ggml_reshape_4d(c, scale, 1, 1, channels, batch);
    bias = ggml_reshape_4d(c, bias, 1, 1, channels, batch);
    return checked_add(c, checked_mul(c, x, scale), bias);
}

ggml_tensor *
alpha_residuals_whc(GraphSession &session, ggml_tensor *x,
                    const std::string &prefix, std::string *error,
                    std::vector<GraphProfilePoint> *profile = nullptr,
                    const std::string &profile_prefix = {}) {
    ggml_context *c = session.context();
    for (int block = 0; block < 2; ++block) {
        const std::string p = prefix + "." + std::to_string(block) + ".net.";
        const std::string detail =
            profile_prefix + ".block" + std::to_string(block);
        ggml_tensor *y = group_norm_affine_whc(
            session, x, p + "0", mask_norm_groups(session), error);
        if (!y)
            return nullptr;
        profile_point(profile, MaskProfileAlphaS1Detail, detail + ".norm0", y);
        y = ggml_gelu_erf(c, y);
        y = conv2d_whc(session, y, p + "2.weight", 1, error);
        if (!y)
            return nullptr;
        profile_point(profile, MaskProfileAlphaS1Detail, detail + ".conv0", y);
        y = group_norm_affine_whc(session, y, p + "3",
                                  mask_norm_groups(session), error);
        if (!y)
            return nullptr;
        profile_point(profile, MaskProfileAlphaS1Detail, detail + ".norm1", y);
        y = ggml_gelu_erf(c, y);
        y = conv2d_whc(session, y, p + "5.weight", 1, error);
        if (!y)
            return nullptr;
        profile_point(profile, MaskProfileAlphaS1Detail, detail + ".conv1", y);
        x = checked_add(c, x, y);
        profile_point(profile, MaskProfileAlphaS1Detail, detail + ".add", x);
    }
    return x;
}

ggml_tensor *alpha_logits_whc(GraphSession &session, ggml_tensor *feature,
                              const std::string &name, std::string *error) {
    ggml_tensor *y = conv2d_whc(session, feature, name + ".weight", 0, error);
    ggml_tensor *bias = session.weight(name + ".bias", error);
    if (!y || !bias || ggml_nelements(bias) != y->ne[2])
        return nullptr;
    return checked_add(
        session.context(), y,
        ggml_reshape_4d(session.context(), bias, 1, 1, y->ne[2], 1));
}
} // namespace

bool build_mask_conv_transpose2x_fixture(
    GraphSession &session, ggml_tensor *input, const std::string &weight_name,
    bool phase_decomposition, ggml_tensor **output, std::string *error) {
    if (!output || !input || input->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(input)) {
        set_error(error, "invalid conv-transpose fixture input");
        return false;
    }
    *output = conv_transpose2x_cwh_impl(session, input, weight_name,
                                        phase_decomposition, error);
    return *output != nullptr;
}

bool build_mask_conv2d_whc_fixture(GraphSession &session, ggml_tensor *input,
                                   ggml_tensor *weight, int padding, int stride,
                                   ggml_tensor **output, std::string *error) {
    if (!output || !input || !weight) {
        set_error(error, "invalid conv2d fixture input");
        return false;
    }
    *output = conv2d_whc_weight(session, input, weight, padding, stride, error);
    return *output != nullptr;
}

bool build_mask_fpn(GraphSession &session, ggml_tensor *memory,
                    const MaskEncoderInputs &spatial, MaskFpnOutput *output,
                    std::string *error) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    const size_t levels =
        static_cast<size_t>(session.model().config.mask.msda_levels);
    if (!output || !memory || memory->type != GGML_TYPE_F32 ||
        memory->ne[0] != mask_hidden || spatial.heights.size() != levels ||
        spatial.widths.size() != levels) {
        set_error(error, "invalid mask FPN input");
        return false;
    }
    ggml_context *c = session.context();
    int64_t offset = 0;
    std::vector<ggml_tensor *> maps;
    for (size_t level = 0; level < levels; ++level) {
        const int64_t width = spatial.widths[level],
                      height = spatial.heights[level], tokens = width * height;
        ggml_tensor *flat = ggml_view_2d(c, memory, mask_hidden, tokens,
                                         memory->nb[1], offset * memory->nb[1]);
        maps.push_back(ggml_reshape_3d(c, flat, mask_hidden, width, height));
        offset += tokens;
    }
    if (offset != memory->ne[1]) {
        set_error(error, "mask FPN spatial shape mismatch");
        return false;
    }
    auto lateral = [&](int level, ggml_tensor **whc) {
        ggml_tensor *flat =
            ggml_reshape_2d(c, maps[level], mask_hidden,
                            maps[level]->ne[1] * maps[level]->ne[2]);
        ggml_tensor *projected = ggml_reshape_3d(
            c,
            checked_mul_mat(
                c,
                ggml_reshape_2d(c,
                                session.weight("wv.mask.pixel.fpn.lateral." +
                                                   std::to_string(level) +
                                                   ".weight",
                                               error),
                                mask_hidden, mask_hidden),
                flat),
            mask_hidden, maps[level]->ne[1], maps[level]->ne[2]);
        if (!projected)
            return false;
        // Convert once to the WHC working layout; every later op in this
        // graph (conv, group norm, add) runs permute-free on it.
        *whc = to_whc(c, projected);
        return *whc != nullptr;
    };
    ggml_tensor *fused = nullptr;
    if (!lateral(static_cast<int>(levels) - 1, &fused))
        return false;
    for (int level = static_cast<int>(levels) - 2, block = 0; level >= 0;
         --level, ++block) {
        ggml_tensor *side = nullptr;
        if (!lateral(level, &side))
            return false;
        fused = checked_add(c, fused, side);
        fused = conv2d_whc(session, fused,
                           "wv.mask.pixel.fpn.output." + std::to_string(block) +
                               ".0.weight",
                           1, error);
        if (!fused)
            return false;
        fused = group_norm_affine_whc(session, fused,
                                      "wv.mask.pixel.fpn.output." +
                                          std::to_string(block) + ".1",
                                      mask_norm_groups(session), error);
        if (!fused)
            return false;
        fused = ggml_gelu_erf(c, fused);
    }
    output->feature = fused;
    if (session.trace_attached()) {
        // fused is already [W,H,C]; the reference hook exports B,C,H,W and
        // FileTraceSink reverses dimensions, so only reshape to 4-D here.
        ggml_tensor *traced = ggml_reshape_4d(c, fused, fused->ne[0],
                                              fused->ne[1], fused->ne[2], 1);
        return session.checkpoint({"mask.pixel.fpn", "mask", "B,C,H,W"}, traced,
                                  error);
    }
    return true;
}

bool build_mask_pixel_fusion(GraphSession &session, ggml_tensor *fpn_feature,
                             ggml_tensor *rgb, ggml_tensor **output,
                             std::string *error) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    const int64_t mask_feature =
        static_cast<int64_t>(session.model().config.mask.feature_size);
    if (!output || !fpn_feature || !rgb || fpn_feature->type != GGML_TYPE_F32 ||
        rgb->type != GGML_TYPE_F32 || fpn_feature->ne[2] != mask_hidden ||
        rgb->ne[0] != 3 || rgb->ne[1] <= 0 || rgb->ne[2] <= 0) {
        set_error(error, "invalid pixel fusion input");
        return false;
    }
    ggml_context *c = session.context();
    // Everything below runs in the WHC working layout: ggml's im2col+GEMM
    // convolution and permute-free group norms. The fpn feature arrives in WHC
    // from build_mask_fpn; the RGB input is converted exactly once and the
    // CWH pixel feature is materialized exactly once at the end.
    ggml_tensor *rgb_whc = to_whc(c, rgb);
    if (!rgb_whc)
        return false;
    ggml_tensor *fpn = pixel_shuffle_stage_whc(
        session, fpn_feature, "wv.mask.pixel.upsample.0", error);
    if (!fpn)
        return false;
    fpn = pixel_shuffle_stage_whc(session, fpn, "wv.mask.pixel.upsample.1",
                                  error);
    if (!fpn)
        return false;
    if (fpn->ne[2] != mask_feature) {
        set_error(error, "invalid pixel fusion upsample channels");
        return false;
    }

    // ShallowImageStem is deliberately evaluated separately: its target size
    // defines the adaptive-pool/bilinear branch in the Python implementation.
    ggml_tensor *cnn = conv2d_whc(
        session, rgb_whc, "wv.mask.pixel.cnn.net.0.weight", 1, 2, error);
    if (!cnn)
        return false;
    cnn = group_norm_affine_whc(session, cnn, "wv.mask.pixel.cnn.net.1",
                                mask_norm_groups(session), error);
    if (!cnn)
        return false;
    cnn = ggml_gelu_erf(c, cnn);
    cnn =
        conv2d_whc(session, cnn, "wv.mask.pixel.cnn.net.3.weight", 1, 1, error);
    if (!cnn)
        return false;
    cnn = group_norm_affine_whc(session, cnn, "wv.mask.pixel.cnn.net.4",
                                mask_norm_groups(session), error);
    if (!cnn)
        return false;
    cnn = ggml_gelu_erf(c, cnn);
    if (cnn->ne[2] != mask_feature) {
        set_error(error, "invalid pixel fusion CNN channels");
        return false;
    }
    if (session.trace_attached()) {
        ggml_tensor *traced =
            ggml_reshape_4d(c, cnn, cnn->ne[0], cnn->ne[1], cnn->ne[2], 1);
        if (!session.checkpoint({"mask.pixel.cnn", "mask", "B,C,H,W"}, traced,
                                error))
            return false;
    }

    const int32_t target_w =
        std::max<int32_t>(1, static_cast<int32_t>(cnn->ne[0] / 2));
    const int32_t target_h =
        std::max<int32_t>(1, static_cast<int32_t>(cnn->ne[1] / 2));
    // The released 1024px path only enlarges the FPN map.  The pool step is
    // nonetheless explicit so a smaller future image cannot silently diverge.
    // The pooling helper is CWH-only, so this rare path pays two permutes.
    const int32_t pool_w =
        std::min<int32_t>(static_cast<int32_t>(fpn->ne[0]), target_w);
    const int32_t pool_h =
        std::min<int32_t>(static_cast<int32_t>(fpn->ne[1]), target_h);
    if (pool_w != fpn->ne[0] || pool_h != fpn->ne[1]) {
        ggml_tensor *cwh = ggml_cont(c, ggml_permute(c, fpn, 1, 2, 0, 3));
        ggml_tensor *pooled = nullptr;
        if (!cwh || !build_adaptive_avg_pool2d(session, cwh, pool_w, pool_h,
                                               &pooled, error))
            return false;
        fpn = to_whc(c, pooled);
        if (!fpn)
            return false;
    }
    if (!resize_whc(session, fpn, target_w, target_h, &fpn, error))
        return false;
    fpn = conv2d_whc(session, fpn, "wv.mask.pixel.fpn_proj.weight", 0, error);
    if (!fpn)
        return false;
    fpn = group_norm_affine_whc(session, fpn, "wv.mask.pixel.smooth.0",
                                mask_norm_groups(session), error);
    if (!fpn)
        return false;
    fpn = ggml_gelu_erf(c, fpn);
    fpn = conv2d_whc(session, fpn, "wv.mask.pixel.smooth.2.weight", 1, error);
    if (!fpn)
        return false;

    ggml_tensor *residual = conv2d_whc(
        session, fpn, "wv.mask.pixel.s2_upsample.conv.weight", 1, error);
    if (!residual)
        return false;
    residual = add_channel_bias_whc(
        session, residual, "wv.mask.pixel.s2_upsample.conv.bias", error);
    if (!residual)
        return false;
    residual = pixel_shuffle2x_whc(session, residual, error);
    if (!residual)
        return false;
    ggml_tensor *base = nullptr;
    if (!resize_whc(session, fpn, static_cast<int32_t>(fpn->ne[0] * 2),
                    static_cast<int32_t>(fpn->ne[1] * 2), &base, error))
        return false;
    if (!ggml_are_same_shape(base, residual)) {
        set_error(error, "invalid residual pixel-shuffle shape");
        return false;
    }
    fpn = checked_add(c, base, residual);
    cnn = conv2d_whc(session, cnn, "wv.mask.pixel.cnn_proj.weight", 0, error);
    if (!cnn)
        return false;
    if (!ggml_are_same_shape(fpn, cnn)) {
        set_error(error, "pixel fusion branch shape mismatch");
        return false;
    }
    ggml_tensor *feature = checked_add(c, fpn, cnn);
    if (session.trace_attached()) {
        ggml_tensor *traced = ggml_reshape_4d(
            c, feature, feature->ne[0], feature->ne[1], feature->ne[2], 1);
        if (!session.checkpoint({"mask.pixel.feature", "mask", "B,C,H,W"},
                                traced, error))
            return false;
    }
    *output = feature;
    return true;
}

bool build_mask_alpha_cascade_whc(
    GraphSession &session, ggml_tensor *rgb, ggml_tensor *pixel_feature,
    ggml_tensor *queries, ggml_tensor *coarse_logits, MaskAlphaOutput *output,
    std::string *error, std::vector<GraphProfilePoint> *profile_points,
    const MaskAlphaShared *shared_alpha) {
    ggml_context *c = session.context();
    const int64_t q = queries->ne[1];
    const int32_t s2w = coarse_logits->ne[0], s2h = coarse_logits->ne[1];
    const int32_t s4w = s2w / 2, s4h = s2h / 2;
    const int32_t s1w = s2w * 2, s1h = s2h * 2;
    auto trace = [&](const std::string &name, ggml_tensor *value) {
        return session.checkpoint({name, "mask", "B,C,H,W"}, value, error);
    };
    auto resize_activation = [&](ggml_tensor *value, int32_t width,
                                 int32_t height, ggml_tensor **resized) {
        return resize_whc(session, value, width, height, resized, error);
    };

    ggml_tensor *rgb_whc = shared_alpha != nullptr ? nullptr : to_whc(c, rgb);
    // pixel_feature is produced in WHC by build_mask_pixel_fusion; consume it
    // directly to avoid a redundant full-resolution CWH<->WHC round trip.
    ggml_tensor *pixel_whc = pixel_feature;
    ggml_tensor *coarse = ggml_reshape_4d(c, coarse_logits, s2w, s2h, 1, q);

    ggml_tensor *semantic = nullptr;
    ggml_tensor *semantic_s4 = nullptr;
    ggml_tensor *rgb_s4 = nullptr;
    if (shared_alpha != nullptr) {
        semantic = shared_alpha->semantic_s2;
        semantic_s4 = shared_alpha->semantic_s4;
        rgb_s4 = shared_alpha->rgb_s4;
    } else {
        semantic = alpha_project_whc(session, pixel_whc,
                                     "wv.mask.alpha.semantic", error);
        if (!semantic)
            return false;
        if (!resize_activation(semantic, s4w, s4h, &semantic_s4))
            return false;
        ggml_tensor *rgb_s4_input = nullptr;
        if (!resize_whc(session, rgb_whc, s4w, s4h, &rgb_s4_input, error))
            return false;
        rgb_s4 = alpha_project_whc(session, rgb_s4_input,
                                   "wv.mask.alpha.rgb_s4", error);
        if (!rgb_s4)
            return false;
    }
    ggml_tensor *coarse_s4 = nullptr;
    if (!resize_whc(session, coarse, s4w, s4h, &coarse_s4, error))
        return false;
    ggml_tensor *coarse_s4_project =
        alpha_project_whc(session, coarse_s4, "wv.mask.alpha.coarse_s4", error);
    if (!coarse_s4_project)
        return false;
    semantic_s4 = expand_batch(c, semantic_s4, q);
    rgb_s4 = expand_batch(c, rgb_s4, q);
    if (!semantic_s4 || !rgb_s4) {
        set_error(error, "invalid alpha s4 batch");
        return false;
    }
    ggml_tensor *feature = checked_concat(c, semantic_s4, rgb_s4, 2);
    feature = checked_concat(c, feature, coarse_s4_project, 2);
    feature =
        conv2d_whc(session, feature, "wv.mask.alpha.fuse_s4.weight", 0, error);
    if (!feature)
        return false;
    feature = alpha_condition_whc(session, feature, queries,
                                  "wv.mask.alpha.query_s4", error);
    if (!feature)
        return false;
    feature =
        alpha_residuals_whc(session, feature, "wv.mask.alpha.blocks_s4", error);
    if (!feature)
        return false;
    ggml_tensor *delta =
        alpha_logits_whc(session, feature, "wv.mask.alpha.output_s4", error);
    if (!delta)
        return false;
    ggml_tensor *logits_s4 = checked_add(c, coarse_s4, delta);
    output->stage_s4 = logits_s4;
    if (!trace("mask.alpha.stage_s4", logits_s4))
        return false;

    ggml_tensor *rgb_s2 = nullptr;
    if (shared_alpha != nullptr) {
        rgb_s2 = shared_alpha->rgb_s2;
    } else {
        ggml_tensor *rgb_s2_input = nullptr;
        if (!resize_whc(session, rgb_whc, s2w, s2h, &rgb_s2_input, error))
            return false;
        rgb_s2 = alpha_project_whc(session, rgb_s2_input,
                                   "wv.mask.alpha.rgb_s2", error);
        if (!rgb_s2)
            return false;
    }
    ggml_tensor *previous_s2_logits = nullptr;
    if (!resize_whc(session, logits_s4, s2w, s2h, &previous_s2_logits, error))
        return false;
    ggml_tensor *previous_s2 =
        alpha_project_whc(session, ggml_sigmoid(c, previous_s2_logits),
                          "wv.mask.alpha.previous_s2", error);
    if (!previous_s2)
        return false;
    ggml_tensor *coarse_s2 =
        alpha_project_whc(session, coarse, "wv.mask.alpha.coarse_s2", error);
    if (!coarse_s2)
        return false;
    ggml_tensor *up = conv_transpose2x_whc(
        session, feature, "wv.mask.alpha.up_s4_to_s2.weight", error);
    if (!up)
        return false;
    ggml_tensor *semantic_s2 = expand_batch(c, semantic, q);
    rgb_s2 = expand_batch(c, rgb_s2, q);
    if (!semantic_s2 || !rgb_s2) {
        set_error(error, "invalid alpha s2 batch");
        return false;
    }
    ggml_tensor *pair0 = checked_concat(c, up, semantic_s2, 2);
    ggml_tensor *pair1 = checked_concat(c, rgb_s2, coarse_s2, 2);
    feature = checked_concat(c, pair0, pair1, 2);
    feature = checked_concat(c, feature, previous_s2, 2);
    feature =
        conv2d_whc(session, feature, "wv.mask.alpha.fuse_s2.weight", 0, error);
    if (!feature)
        return false;
    feature = alpha_condition_whc(session, feature, queries,
                                  "wv.mask.alpha.query_s2", error);
    if (!feature)
        return false;
    feature =
        alpha_residuals_whc(session, feature, "wv.mask.alpha.blocks_s2", error);
    if (!feature)
        return false;
    delta =
        alpha_logits_whc(session, feature, "wv.mask.alpha.output_s2", error);
    if (!delta)
        return false;
    ggml_tensor *logits_s2 = checked_add(c, coarse, delta);
    output->stage_s2 = logits_s2;
    if (!trace("mask.alpha.stage_s2", logits_s2))
        return false;

    ggml_tensor *semantic_s1 = nullptr;
    ggml_tensor *rgb_s1 = nullptr;
    if (shared_alpha != nullptr) {
        semantic_s1 = shared_alpha->semantic_s1;
        rgb_s1 = shared_alpha->rgb_s1;
    } else {
        if (!resize_activation(semantic, s1w, s1h, &semantic_s1))
            return false;
        profile_point(profile_points, MaskProfileAlphaS1Detail,
                      "mask.detail.s1.semantic_resize", semantic_s1);
        rgb_s1 = alpha_project_whc(session, rgb_whc, "wv.mask.alpha.rgb_s1",
                                   error, profile_points, "mask.detail.s1.rgb");
        if (!rgb_s1)
            return false;
    }
    semantic_s1 = expand_batch(c, semantic_s1, q);
    if (!semantic_s1)
        return false;
    rgb_s1 = expand_batch(c, rgb_s1, q);
    if (!rgb_s1)
        return false;
    ggml_tensor *previous_s1_logits = nullptr;
    if (!resize_whc(session, logits_s2, s1w, s1h, &previous_s1_logits, error))
        return false;
    profile_point(profile_points, MaskProfileAlphaS1Detail,
                  "mask.detail.s1.previous_resize", previous_s1_logits);
    ggml_tensor *previous_s1 =
        alpha_project_whc(session, ggml_sigmoid(c, previous_s1_logits),
                          "wv.mask.alpha.previous_s1", error, profile_points,
                          "mask.detail.s1.previous");
    if (!previous_s1)
        return false;
    ggml_tensor *up_s1 = conv_transpose2x_whc(
        session, feature, "wv.mask.alpha.up_s2_to_s1.weight", error);
    if (!up_s1)
        return false;
    profile_point(profile_points, MaskProfileAlphaS1Detail, "mask.detail.s1.up",
                  up_s1);
    ggml_tensor *s1_pair0 = checked_concat(c, up_s1, semantic_s1, 2);
    ggml_tensor *s1_pair1 = checked_concat(c, rgb_s1, previous_s1, 2);
    ggml_tensor *feature_s1 = checked_concat(c, s1_pair0, s1_pair1, 2);
    profile_point(profile_points, MaskProfileAlphaS1Detail,
                  "mask.detail.s1.concat", feature_s1);
    feature_s1 = conv2d_whc(session, feature_s1, "wv.mask.alpha.fuse_s1.weight",
                            0, error);
    if (!feature_s1)
        return false;
    feature_s1 = alpha_condition_whc(session, feature_s1, queries,
                                     "wv.mask.alpha.query_s1", error);
    if (!feature_s1)
        return false;
    profile_point(profile_points, MaskProfileAlphaS1Detail,
                  "mask.detail.s1.condition", feature_s1);
    feature_s1 =
        alpha_residuals_whc(session, feature_s1, "wv.mask.alpha.blocks_s1",
                            error, profile_points, "mask.detail.s1.residual");
    if (!feature_s1 || !trace("mask.alpha.blocks_s1", feature_s1))
        return false;
    ggml_tensor *delta_s1 =
        alpha_logits_whc(session, feature_s1, "wv.mask.alpha.output_s1", error);
    if (!delta_s1 || !trace("mask.alpha.delta_s1", delta_s1))
        return false;
    ggml_tensor *logits_s1 = checked_add(c, previous_s1_logits, delta_s1);
    if (!trace("mask.alpha.previous_s1_logits", previous_s1_logits))
        return false;
    output->logits_s1 = ggml_reshape_3d(c, logits_s1, s1w, s1h, q);
    output->logits_s2 = ggml_reshape_3d(c, logits_s2, s2w, s2h, q);
    return session.checkpoint({"mask.alpha.logits_s1", "mask", "objects,H,W"},
                              output->logits_s1, error) &&
           session.checkpoint({"mask.alpha.logits_s2", "mask", "objects,H,W"},
                              output->logits_s2, error);
}

bool build_mask_alpha_shared(GraphSession &session, ggml_tensor *rgb,
                             ggml_tensor *pixel_feature,
                             MaskAlphaShared *output, std::string *error) {
    if (!output || !rgb || !pixel_feature || rgb->type != GGML_TYPE_F32 ||
        pixel_feature->type != GGML_TYPE_F32 || rgb->ne[0] != 3 ||
        pixel_feature->ne[2] !=
            static_cast<int64_t>(session.model().config.mask.feature_size) ||
        pixel_feature->ne[0] * 2 != rgb->ne[1] ||
        pixel_feature->ne[1] * 2 != rgb->ne[2]) {
        set_error(error, "invalid alpha shared input");
        return false;
    }
    ggml_context *c = session.context();
    const int32_t s2w = static_cast<int32_t>(pixel_feature->ne[0]);
    const int32_t s2h = static_cast<int32_t>(pixel_feature->ne[1]);
    const int32_t s4w = s2w / 2;
    const int32_t s4h = s2h / 2;
    const int32_t s1w = static_cast<int32_t>(rgb->ne[1]);
    const int32_t s1h = static_cast<int32_t>(rgb->ne[2]);
    ggml_tensor *rgb_whc = to_whc(c, rgb);

    ggml_tensor *semantic = alpha_project_whc(session, pixel_feature,
                                              "wv.mask.alpha.semantic", error);
    if (!semantic)
        return false;
    ggml_tensor *semantic_s4 = nullptr;
    if (!resize_whc(session, semantic, s4w, s4h, &semantic_s4, error))
        return false;
    ggml_tensor *semantic_s1 = nullptr;
    if (!resize_whc(session, semantic, s1w, s1h, &semantic_s1, error))
        return false;
    ggml_tensor *rgb_s4_input = nullptr;
    if (!resize_whc(session, rgb_whc, s4w, s4h, &rgb_s4_input, error))
        return false;
    ggml_tensor *rgb_s4 =
        alpha_project_whc(session, rgb_s4_input, "wv.mask.alpha.rgb_s4", error);
    if (!rgb_s4)
        return false;
    ggml_tensor *rgb_s2_input = nullptr;
    if (!resize_whc(session, rgb_whc, s2w, s2h, &rgb_s2_input, error))
        return false;
    ggml_tensor *rgb_s2 =
        alpha_project_whc(session, rgb_s2_input, "wv.mask.alpha.rgb_s2", error);
    if (!rgb_s2)
        return false;
    ggml_tensor *rgb_s1 =
        alpha_project_whc(session, rgb_whc, "wv.mask.alpha.rgb_s1", error);
    if (!rgb_s1)
        return false;

    output->semantic_s4 = semantic_s4;
    output->semantic_s2 = semantic;
    output->semantic_s1 = semantic_s1;
    output->rgb_s4 = rgb_s4;
    output->rgb_s2 = rgb_s2;
    output->rgb_s1 = rgb_s1;
    return true;
}

bool build_mask_alpha_cascade(GraphSession &session, ggml_tensor *rgb,
                              ggml_tensor *pixel_feature, ggml_tensor *queries,
                              ggml_tensor *coarse_logits,
                              MaskAlphaOutput *output, std::string *error,
                              std::vector<GraphProfilePoint> *profile_points,
                              const MaskAlphaShared *shared_alpha) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    if (!output || !pixel_feature || !queries || !coarse_logits ||
        pixel_feature->type != GGML_TYPE_F32 ||
        queries->type != GGML_TYPE_F32 ||
        coarse_logits->type != GGML_TYPE_F32 ||
        pixel_feature->ne[2] !=
            static_cast<int64_t>(session.model().config.mask.feature_size) ||
        queries->ne[0] != mask_hidden ||
        coarse_logits->ne[2] != queries->ne[1]) {
        set_error(error, "invalid alpha cascade input");
        return false;
    }
    if (shared_alpha != nullptr) {
        if (!shared_alpha->semantic_s4 || !shared_alpha->semantic_s2 ||
            !shared_alpha->semantic_s1 || !shared_alpha->rgb_s4 ||
            !shared_alpha->rgb_s2 || !shared_alpha->rgb_s1 ||
            shared_alpha->semantic_s1->ne[0] != coarse_logits->ne[0] * 2 ||
            shared_alpha->semantic_s1->ne[1] != coarse_logits->ne[1] * 2) {
            set_error(error, "invalid shared alpha cascade input");
            return false;
        }
    } else if (!rgb || rgb->type != GGML_TYPE_F32 || rgb->ne[0] != 3 ||
               coarse_logits->ne[0] * 2 != rgb->ne[1] ||
               coarse_logits->ne[1] * 2 != rgb->ne[2]) {
        set_error(error, "invalid alpha cascade input");
        return false;
    }
    // Both backends use the WHC cascade: ggml's im2col+GEMM convolution and
    // permute-free group norms are substantially faster than the retired CWH
    // direct path on CPU and Metal alike. Verified equivalent on the seabreeze
    // regression input (token sequence, box coordinates, and mask/alpha).
    return build_mask_alpha_cascade_whc(session, rgb, pixel_feature, queries,
                                        coarse_logits, output, error,
                                        profile_points, shared_alpha);
}

bool build_dynamic_mask_predictor(GraphSession &session, ggml_tensor *queries,
                                  ggml_tensor *pixel_feature,
                                  ggml_tensor **output, std::string *error) {
    const int64_t mask_hidden =
        static_cast<int64_t>(session.model().config.mask.hidden_size);
    const int64_t mask_feature =
        static_cast<int64_t>(session.model().config.mask.feature_size);
    if (!output || !queries || !pixel_feature ||
        queries->type != GGML_TYPE_F32 ||
        pixel_feature->type != GGML_TYPE_F32 || queries->ne[0] != mask_hidden ||
        pixel_feature->ne[2] != mask_feature || pixel_feature->ne[0] <= 0 ||
        pixel_feature->ne[1] <= 0) {
        set_error(error, "invalid dynamic mask predictor input");
        return false;
    }
    ggml_context *c = session.context();
    const int64_t count = queries->ne[1], width = pixel_feature->ne[0],
                  height = pixel_feature->ne[1], pixels = width * height;
    ggml_tensor *params =
        linear(session, queries, "wv.mask.mask_predictor.kernel.0",
               static_cast<int>(mask_hidden), error);
    if (!params)
        return false;
    params = ggml_gelu_erf(c, params);
    // One dynamic kernel per object plus a scalar bias: feature_size + 1
    // outputs per query.
    params = linear(session, params, "wv.mask.mask_predictor.kernel.2",
                    static_cast<int>(mask_feature) + 1, error);
    if (!params)
        return false;
    ggml_tensor *kernels =
        ggml_view_2d(c, params, mask_feature, count, params->nb[1], 0);
    ggml_tensor *bias = ggml_view_2d(c, params, 1, count, params->nb[1],
                                     mask_feature * sizeof(float));
    // The bias view is a strided [1, count] column: one scalar per object.
    // ggml_add broadcasts b onto a only when every a dim is divisible by the
    // matching b dim, so [1, count] against [count, pixels] asserts (or, when
    // pixels % count == 0, silently adds object (p % count)'s bias to pixel
    // p). Transpose to a contiguous [count, 1] so each object row receives
    // exactly its own scalar bias along the pixels dimension.
    bias = ggml_cont(c, ggml_transpose(c, bias));
    // pixel_feature arrives in WHC [W,H,C]; the per-pixel GEMM contraction
    // needs channels in the fastest ggml dimension, so convert to CWH once.
    // This is the only full-resolution transpose on the pixel feature path:
    // build_mask_pixel_fusion returns WHC and build_mask_alpha_cascade
    // consumes it directly, avoiding the previous CWH<->WHC round trip.
    ggml_tensor *pixel_cwh =
        ggml_cont(c, ggml_permute(c, pixel_feature, 1, 2, 0, 3));
    ggml_tensor *flat = ggml_reshape_2d(c, pixel_cwh, mask_feature, pixels);
    ggml_tensor *logits = checked_mul_mat(c, kernels, flat);
    logits = checked_add(c, logits, bias); // [objects,pixels]
    logits = ggml_reshape_3d(c, logits, count, width, height);
    *output =
        ggml_cont(c, ggml_permute(c, logits, 2, 0, 1, 3)); // [W,H,objects]
    return session.checkpoint({"mask.coarse_logits", "mask", "objects,H,W"},
                              *output, error);
}
} // namespace wevision
