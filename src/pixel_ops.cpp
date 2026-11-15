#include "pixel_ops.hpp"
#include "graph_ops.hpp"

#include "ggml.h"

#include <algorithm>
#include <vector>

namespace wevision {
namespace {
bool valid(ggml_tensor *x) {
    return x && x->type == GGML_TYPE_F32 && x->ne[0] > 0 && x->ne[1] > 0 &&
           x->ne[2] > 0 && ggml_is_contiguous(x);
}

bool adaptive_pool_rows(GraphSession &session, ggml_tensor *input,
                        int32_t target_rows, ggml_tensor **output,
                        std::string *error) {
    const int32_t source_rows = static_cast<int32_t>(input->ne[1]);
    if (target_rows <= 0 || target_rows > source_rows) {
        *error = "invalid adaptive average pool axis";
        return false;
    }
    if (target_rows == source_rows) {
        *output = input;
        return true;
    }
    int32_t max_bin = 0;
    for (int32_t row = 0; row < target_rows; ++row) {
        const int32_t begin = static_cast<int32_t>(
            (static_cast<int64_t>(row) * source_rows) / target_rows);
        const int32_t end = static_cast<int32_t>(
            (static_cast<int64_t>(row + 1) * source_rows + target_rows - 1) /
            target_rows);
        max_bin = std::max(max_bin, end - begin);
    }
    ggml_context *ctx = session.context();
    ggml_tensor *sum = nullptr;
    const int64_t planes = input->ne[2] * input->ne[3];
    for (int32_t slot = 0; slot < max_bin; ++slot) {
        std::vector<int32_t> indices(static_cast<size_t>(target_rows) * planes);
        std::vector<float> weights(indices.size());
        for (int64_t plane = 0; plane < planes; ++plane) {
            for (int32_t row = 0; row < target_rows; ++row) {
                const int32_t begin = static_cast<int32_t>(
                    (static_cast<int64_t>(row) * source_rows) / target_rows);
                const int32_t end = static_cast<int32_t>(
                    (static_cast<int64_t>(row + 1) * source_rows + target_rows -
                     1) /
                    target_rows);
                const int32_t count = end - begin;
                const size_t index =
                    static_cast<size_t>(plane) * target_rows + row;
                indices[index] = begin + std::min(slot, count - 1);
                weights[index] = slot < count ? 1.0F / count : 0.0F;
            }
        }
        ggml_tensor *row_ids = ggml_new_tensor_3d(
            ctx, GGML_TYPE_I32, target_rows, input->ne[2], input->ne[3]);
        ggml_tensor *row_weights = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, 1, target_rows, input->ne[2], input->ne[3]);
        if (!session.set_input(row_ids, indices.data(), ggml_nbytes(row_ids),
                               error) ||
            !session.set_input(row_weights, weights.data(),
                               ggml_nbytes(row_weights), error))
            return false;
        ggml_tensor *term =
            checked_mul(ctx, ggml_get_rows(ctx, input, row_ids), row_weights);
        sum = sum == nullptr ? term : checked_add(ctx, sum, term);
    }
    *output = sum;
    return true;
}
} // namespace

bool build_pixel_shuffle2x(GraphSession &session, ggml_tensor *input,
                           ggml_tensor **output, std::string *error) {
    if (!output || !valid(input) || input->ne[0] % 4) {
        *error = "invalid pixel shuffle input";
        return false;
    }
    ggml_context *ctx = session.context();
    const int64_t channels = input->ne[0] / 4;
    const int64_t width = input->ne[1];
    const int64_t height = input->ne[2];
    const int64_t batch = input->ne[3];

    // Channel order is [dx,dy,C]. Interleave dx with W first, then dy with H.
    // Keeping each permutation contiguous makes the graph backend-neutral and
    // lets Metal execute the shuffle without a custom CPU callback.
    ggml_tensor *horizontal =
        ggml_reshape_4d(ctx, input, 2, 2 * channels, width, height * batch);
    horizontal = ggml_cont(ctx, ggml_permute(ctx, horizontal, 1, 0, 2, 3));
    horizontal = ggml_reshape_4d(ctx, horizontal, 2 * channels, 2 * width,
                                 height, batch);

    ggml_tensor *vertical = ggml_reshape_4d(ctx, horizontal, 2, channels,
                                            2 * width, height * batch);
    vertical = ggml_cont(ctx, ggml_permute(ctx, vertical, 2, 0, 1, 3));
    *output =
        ggml_reshape_4d(ctx, vertical, channels, 2 * width, 2 * height, batch);
    return true;
}

bool build_bilinear_resize(GraphSession &session, ggml_tensor *input,
                           int32_t width, int32_t height, ggml_tensor **output,
                           std::string *error) {
    if (!output || !valid(input) || width <= 0 || height <= 0) {
        *error = "invalid bilinear resize input";
        return false;
    }
    // ggml's bilinear interpolation operates on dimensions 0 and 1. Our image
    // tensors are C,W,H,N, so make W,H contiguous first and restore the public
    // layout afterwards. This maps to GGML_OP_UPSCALE on both CPU and Metal.
    ggml_tensor *whcn = ggml_cont(
        session.context(), ggml_permute(session.context(), input, 2, 0, 1, 3));
    ggml_tensor *resized =
        ggml_interpolate(session.context(), whcn, width, height, input->ne[0],
                         input->ne[3], GGML_SCALE_MODE_BILINEAR);
    *output = ggml_cont(session.context(),
                        ggml_permute(session.context(), resized, 1, 2, 0, 3));
    return true;
}

bool build_adaptive_avg_pool2d(GraphSession &session, ggml_tensor *input,
                               int32_t width, int32_t height,
                               ggml_tensor **output, std::string *error) {
    if (!output || !valid(input) || width <= 0 || height <= 0 ||
        width > input->ne[1] || height > input->ne[2]) {
        *error = "invalid adaptive average pool input";
        return false;
    }
    ggml_tensor *pooled_width = nullptr;
    if (!adaptive_pool_rows(session, input, width, &pooled_width, error))
        return false;
    ggml_tensor *transposed =
        ggml_cont(session.context(),
                  ggml_permute(session.context(), pooled_width, 0, 2, 1, 3));
    ggml_tensor *pooled_height = nullptr;
    if (!adaptive_pool_rows(session, transposed, height, &pooled_height, error))
        return false;
    *output =
        ggml_cont(session.context(),
                  ggml_permute(session.context(), pooled_height, 0, 2, 1, 3));
    return true;
}
} // namespace wevision
