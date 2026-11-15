#include "mask_ops.hpp"

#include "graph_ops.hpp"
#include "pixel_ops.hpp"
#include "profile.hpp"

#include "ggml.h"

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wevision {
namespace mask_detail {
namespace {

// Cached once per process: forcing the exact F32 direct convolution instead
// of the faster lowered path is a debug/regression knob, not a runtime one.
bool mask_s1_f32_direct_enabled() {
    static const bool enabled =
        DebugOptions::from_environment().mask_s1_f32_direct;
    return enabled;
}

// Reorders a 4x4 ConvTranspose2d(stride=2, padding=1) kernel into four 3x3
// spatial-phase convolution kernels [3, 3, IC, 4*OC].
ggml_tensor *build_conv_transpose2x_phase_weight(GraphSession &session,
                                                 ggml_tensor *w,
                                                 std::string *error) {
    const int64_t input_channels = w->ne[3];
    const int64_t output_channels = w->ne[2];
    static std::mutex cache_mutex;
    static std::unordered_map<const ggml_tensor *, std::vector<float>> cache;
    std::vector<float> cached_phase;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(w);
        if (found != cache.end())
            cached_phase = found->second;
    }
    if (cached_phase.empty()) {
        std::vector<float> source(static_cast<size_t>(ggml_nelements(w)));
        if (!session.copy_to_host(w, source.data(), ggml_nbytes(w), error))
            return nullptr;
        cached_phase.assign(
            static_cast<size_t>(3 * 3 * input_channels * 4 * output_channels),
            0.0F);
        for (int64_t ic = 0; ic < input_channels; ++ic) {
            for (int64_t oc = 0; oc < output_channels; ++oc) {
                for (int py = 0; py < 2; ++py) {
                    for (int px = 0; px < 2; ++px) {
                        const int64_t out = px + 2 * py + 4 * oc;
                        for (int ky = py ? 0 : 1; ky < 4; ky += 2) {
                            for (int kx = px ? 0 : 1; kx < 4; kx += 2) {
                                const int qy = (py + 1 - ky) / 2 + 1;
                                const int qx = (px + 1 - kx) / 2 + 1;
                                const size_t src = static_cast<size_t>(
                                    kx +
                                    4 * (ky + 4 * (oc + output_channels * ic)));
                                const size_t dst = static_cast<size_t>(
                                    qx +
                                    3 * (qy +
                                         3 * (ic + input_channels * out)));
                                cached_phase[dst] = source[src];
                            }
                        }
                    }
                }
            }
        }
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(w, cached_phase);
    }
    ggml_tensor *phase_weight =
        ggml_new_tensor_4d(session.context(), GGML_TYPE_F32, 3, 3,
                           input_channels, 4 * output_channels);
    if (!session.set_input(phase_weight, cached_phase.data(),
                           ggml_nbytes(phase_weight), error))
        return nullptr;
    return phase_weight;
}

} // namespace

void profile_point(std::vector<GraphProfilePoint> *points, MaskProfileTag tag,
                   std::string name, ggml_tensor *tensor) {
    if (points)
        points->push_back({std::move(name), tensor, tag});
}

ggml_tensor *linear(GraphSession &session, ggml_tensor *x,
                    const std::string &name, int out, std::string *error) {
    return linear_parameters(session, x, name + ".weight", name + ".bias", out,
                             error);
}

ggml_tensor *linear_parameters(GraphSession &session, ggml_tensor *x,
                               const std::string &weight_name,
                               const std::string &bias_name, int out,
                               std::string *error) {
    ggml_tensor *w = session.weight(weight_name, error);
    ggml_tensor *b = session.weight(bias_name, error);
    if (!w || !b)
        return nullptr;
    if (w->ne[0] != x->ne[0] || w->ne[1] != out || ggml_nelements(b) != out) {
        set_error(error, "invalid linear weight shape: " + weight_name);
        return nullptr;
    }
    ggml_context *c = session.context();
    ggml_tensor *y = checked_mul_mat(c, w, x);
    return checked_add(c, y, ggml_reshape_2d(c, b, out, 1));
}

int mask_norm_groups(const GraphSession &session) {
    return static_cast<int>(session.model().config.mask.norm_groups);
}

ggml_tensor *layer_norm(GraphSession &session, ggml_tensor *x,
                        const std::string &name, std::string *error) {
    ggml_tensor *w = session.weight(name + ".weight", error);
    ggml_tensor *b = session.weight(name + ".bias", error);
    if (!w || !b)
        return nullptr;
    if (ggml_nelements(w) != x->ne[0] || ggml_nelements(b) != x->ne[0]) {
        set_error(error, "invalid layer norm parameter shape: " + name);
        return nullptr;
    }
    ggml_context *c = session.context();
    ggml_tensor *y = ggml_norm(c, x, session.model().config.mask.norm_eps);
    y = checked_mul(c, y, ggml_reshape_2d(c, w, x->ne[0], 1));
    return checked_add(c, y, ggml_reshape_2d(c, b, x->ne[0], 1));
}

ggml_tensor *conv2d_whc_weight(GraphSession &session, ggml_tensor *x,
                               ggml_tensor *w, int padding, int stride,
                               std::string *error) {
    if (!x || !w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        w->ne[2] != x->ne[2]) {
        set_error(error, "invalid conv2d input or weight shape");
        return nullptr;
    }
    ggml_context *c = session.context();
    // Exact F32 fallback for precision audits and backend regressions.
    if (mask_s1_f32_direct_enabled())
        return checked_conv_2d_direct(c, w, x, stride, stride, padding, padding,
                                      1, 1);
    // Same lowering as ggml_conv_2d (im2col -> GEMM with a=cols, b=weight) but
    // skips the trailing permute+cont when N == 1.
    if (x->ne[3] != 1)
        return checked_conv_2d(c, w, x, stride, stride, padding, padding, 1, 1);
    ggml_tensor *cols = ggml_im2col(c, w, x, stride, stride, padding, padding,
                                    1, 1, true, GGML_TYPE_F16);
    ggml_tensor *out = checked_mul_mat(
        c,
        ggml_reshape_2d(c, cols, cols->ne[0],
                        cols->ne[3] * cols->ne[2] * cols->ne[1]),
        ggml_reshape_2d(c, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    return ggml_reshape_4d(c, out, cols->ne[1], cols->ne[2], w->ne[3], 1);
}

ggml_tensor *conv2d_whc(GraphSession &session, ggml_tensor *x,
                        const std::string &weight_name, int padding, int stride,
                        std::string *error) {
    ggml_tensor *w = session.weight(weight_name, error);
    if (!w)
        return nullptr;
    return conv2d_whc_weight(session, x, w, padding, stride, error);
}

ggml_tensor *conv2d_whc(GraphSession &session, ggml_tensor *x,
                        const std::string &weight_name, int padding,
                        std::string *error) {
    return conv2d_whc(session, x, weight_name, padding, 1, error);
}

ggml_tensor *add_channel_bias_whc(GraphSession &session, ggml_tensor *x,
                                  const std::string &bias_name,
                                  std::string *error) {
    ggml_tensor *bias = session.weight(bias_name, error);
    if (!bias)
        return nullptr;
    if (bias->type != GGML_TYPE_F32 || ggml_nelements(bias) != x->ne[2]) {
        set_error(error, "invalid channel bias shape: " + bias_name);
        return nullptr;
    }
    ggml_context *c = session.context();
    return checked_add(c, x, ggml_reshape_4d(c, bias, 1, 1, x->ne[2], 1));
}

ggml_tensor *group_norm_affine_whc(GraphSession &session, ggml_tensor *x,
                                   const std::string &name, int groups,
                                   std::string *error) {
    ggml_tensor *scale = session.weight(name + ".weight", error);
    ggml_tensor *bias = session.weight(name + ".bias", error);
    if (!scale || !bias)
        return nullptr;
    if (x->type != GGML_TYPE_F32 || x->ne[2] != ggml_nelements(scale) ||
        ggml_nelements(scale) != ggml_nelements(bias)) {
        set_error(error, "invalid group norm parameter shape: " + name);
        return nullptr;
    }
    ggml_context *c = session.context();
    const int64_t channels_per_group = x->ne[2] / groups;
    ggml_tensor *ss = ggml_reshape_4d(c, scale, 1, 1, x->ne[2], 1);
    ggml_tensor *bb = ggml_reshape_4d(c, bias, 1, 1, x->ne[2], 1);
    if (groups > 0 && x->ne[2] % groups == 0 && ggml_is_contiguous(x)) {
        const int64_t group_elems = x->ne[0] * x->ne[1] * channels_per_group;
        ggml_tensor *grouped =
            ggml_reshape_4d(c, x, group_elems, groups, x->ne[3], 1);
        ggml_tensor *normed =
            ggml_norm(c, grouped, session.model().config.mask.norm_eps);
        ggml_tensor *y =
            ggml_reshape_4d(c, normed, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        return checked_add(c, checked_mul(c, y, ss), bb);
    }
    ggml_tensor *y =
        ggml_group_norm(c, x, groups, session.model().config.mask.norm_eps);
    return checked_add(c, checked_mul(c, y, ss), bb);
}

ggml_tensor *expand_batch(ggml_context *c, ggml_tensor *x, int64_t batch) {
    if (x->ne[3] == batch)
        return x;
    if (x->ne[3] != 1)
        return nullptr;
    ggml_tensor *target =
        ggml_new_tensor_4d(c, x->type, x->ne[0], x->ne[1], x->ne[2], batch);
    return checked_repeat(c, x, target);
}

ggml_tensor *to_whc(ggml_context *c, ggml_tensor *x) {
    return ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3));
}

bool resize_whc(GraphSession &session, ggml_tensor *x, int32_t width,
                int32_t height, ggml_tensor **output, std::string *error) {
    if (!output || !x || x->type != GGML_TYPE_F32 || width <= 0 ||
        height <= 0 || !ggml_is_contiguous(x)) {
        set_error(error, "invalid WHC bilinear resize input");
        return false;
    }
    ggml_tensor *resized =
        ggml_interpolate(session.context(), x, width, height, x->ne[2],
                         x->ne[3], GGML_SCALE_MODE_BILINEAR);
    *output = resized;
    return true;
}

ggml_tensor *pixel_shuffle2x_whc(GraphSession &session, ggml_tensor *x,
                                 std::string *error) {
    if (!x || x->type != GGML_TYPE_F32 || x->ne[2] % 4 != 0 ||
        !ggml_is_contiguous(x)) {
        set_error(error, "invalid WHC pixel shuffle input");
        return nullptr;
    }
    ggml_context *c = session.context();
    const int64_t width = x->ne[0];
    const int64_t height = x->ne[1];
    const int64_t channels = x->ne[2] / 4;
    const int64_t batch = x->ne[3];

    // Channel order is [dx, dy, C]. Interleave dx with W, then dy with H
    // directly in W,H,C layout to avoid WHC<->CWH round-trip copies.
    ggml_tensor *horizontal =
        ggml_reshape_4d(c, x, width, height, 2, 2 * channels * batch);
    horizontal = ggml_cont(c, ggml_permute(c, horizontal, 1, 2, 0, 3));
    horizontal = ggml_reshape_4d(c, horizontal, 2 * width, height, 2,
                                 channels * batch);

    ggml_tensor *vertical =
        ggml_cont(c, ggml_permute(c, horizontal, 0, 2, 1, 3));
    return ggml_reshape_4d(c, vertical, 2 * width, 2 * height, channels,
                           batch);
}

ggml_tensor *conv_transpose2x_whc(GraphSession &session, ggml_tensor *x,
                                  const std::string &weight_name,
                                  std::string *error) {
    ggml_tensor *w = session.weight(weight_name, error);
    if (!w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != 4 ||
        w->ne[3] != x->ne[2] || x->type != GGML_TYPE_F32)
        return nullptr;
    ggml_tensor *phase_weight =
        build_conv_transpose2x_phase_weight(session, w, error);
    if (!phase_weight)
        return nullptr;
    ggml_tensor *phases =
        conv2d_whc_weight(session, x, phase_weight, 1, 1, error);
    if (!phases)
        return nullptr;
    return pixel_shuffle2x_whc(session, phases, error);
}

ggml_tensor *conv_transpose2x_cwh_impl(GraphSession &session, ggml_tensor *x,
                                       const std::string &weight_name,
                                       bool phase_decomposition,
                                       std::string *error) {
    ggml_tensor *w = session.weight(weight_name, error);
    if (!w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != 4 ||
        w->ne[3] != x->ne[0])
        return nullptr;
    ggml_context *c = session.context();
    if (phase_decomposition) {
        ggml_tensor *phase_weight =
            build_conv_transpose2x_phase_weight(session, w, error);
        if (!phase_weight)
            return nullptr;
        ggml_tensor *whc = ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3));
        ggml_tensor *phases =
            checked_conv_2d_direct(c, phase_weight, whc, 1, 1, 1, 1, 1, 1);
        phases = ggml_cont(c, ggml_permute(c, phases, 1, 2, 0, 3));
        ggml_tensor *shuffled = nullptr;
        return build_pixel_shuffle2x(session, phases, &shuffled, error)
                   ? shuffled
                   : nullptr;
    }
    ggml_tensor *whc = ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3));
    ggml_tensor *raw = checked_conv_transpose_2d_p0(c, w, whc, 2);
    // ggml exposes only p=0; PyTorch ConvTranspose2d(k=4,s=2,p=1) is its
    // centered crop.
    ggml_tensor *cropped = ggml_view_4d(
        c, raw, 2 * x->ne[1], 2 * x->ne[2], raw->ne[2], raw->ne[3], raw->nb[1],
        raw->nb[2], raw->nb[3], raw->nb[1] + raw->nb[0]);
    return ggml_cont(c, ggml_permute(c, cropped, 1, 2, 0, 3));
}

} // namespace mask_detail
} // namespace wevision
