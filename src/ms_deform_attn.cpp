#include "ms_deform_attn.hpp"

#include "graph_ops.hpp"

#include "ggml.h"

#include <vector>

namespace wevision {
namespace {

ggml_tensor *level_slice(ggml_context *ctx, ggml_tensor *tensor, int64_t level,
                         int64_t points) {
    return ggml_cont(
        ctx, ggml_view_3d(ctx, tensor, points, tensor->ne[1], tensor->ne[2],
                          tensor->nb[1], tensor->nb[2],
                          static_cast<size_t>(level * points) * tensor->nb[0]));
}

ggml_tensor *coordinate_validity(ggml_context *ctx, ggml_tensor *coordinate,
                                 float extent) {
    // Coordinates are integer-valued here. The half-unit offsets keep exact
    // boundary values away from STEP's strict x > 0 threshold.
    ggml_tensor *lower =
        ggml_step(ctx, ggml_scale_bias(ctx, coordinate, 1.0F, 0.5F));
    ggml_tensor *upper =
        ggml_step(ctx, ggml_scale_bias(ctx, coordinate, -1.0F, extent - 0.5F));
    return checked_mul(ctx, lower, upper);
}

} // namespace

bool build_ms_deform_attn_sampler(GraphSession &session, ggml_tensor *anchor,
                                  ggml_tensor *value, ggml_tensor *locations,
                                  ggml_tensor *weights,
                                  const MSDeformAttnConfig &config,
                                  ggml_tensor **output, std::string *error) {
    if (!anchor || !value || !locations || !weights || !output ||
        config.head_dim <= 0 || config.heads <= 0 || config.levels <= 0 ||
        config.levels > 8 || config.points <= 0 ||
        config.heights.size() != static_cast<size_t>(config.levels) ||
        config.widths.size() != static_cast<size_t>(config.levels) ||
        anchor->type != GGML_TYPE_F32 || value->type != GGML_TYPE_F32 ||
        locations->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(anchor) || !ggml_is_contiguous(value) ||
        !ggml_is_contiguous(locations) || !ggml_is_contiguous(weights)) {
        *error = "invalid MSDeformAttn sampler inputs";
        return false;
    }

    const int64_t per_head =
        static_cast<int64_t>(config.points) * config.levels;
    const int64_t queries = anchor->ne[1];
    int64_t total_tokens = 0;
    for (int level = 0; level < config.levels; ++level) {
        if (config.heights[level] <= 0 || config.widths[level] <= 0) {
            *error = "invalid MSDeformAttn spatial shape";
            return false;
        }
        total_tokens +=
            static_cast<int64_t>(config.heights[level]) * config.widths[level];
    }
    if (anchor->ne[0] != static_cast<int64_t>(config.head_dim) * config.heads ||
        value->ne[0] != config.head_dim || value->ne[1] != config.heads ||
        value->ne[2] != total_tokens || locations->ne[0] != 2 ||
        locations->ne[1] != per_head || locations->ne[2] != config.heads ||
        locations->ne[3] != queries || weights->ne[0] != per_head ||
        weights->ne[1] != config.heads || weights->ne[2] != queries) {
        *error = "MSDeformAttn tensor shape mismatch";
        return false;
    }

    ggml_context *ctx = session.context();
    const int64_t samples =
        static_cast<int64_t>(config.points) * config.heads * queries;
    ggml_tensor *value_rows = ggml_reshape_2d(ctx, value, config.head_dim,
                                              total_tokens * config.heads);

    // Move xy to the last dimension so each component and each level can be
    // represented by a contiguous [points,heads,queries] tensor.
    ggml_tensor *locations_xy_last =
        ggml_cont(ctx, ggml_permute(ctx, locations, 3, 0, 1, 2));
    const size_t component_bytes =
        static_cast<size_t>(per_head) * config.heads * queries * sizeof(float);
    ggml_tensor *x_all = ggml_view_3d(
        ctx, locations_xy_last, per_head, config.heads, queries,
        per_head * sizeof(float), per_head * config.heads * sizeof(float), 0);
    ggml_tensor *y_all =
        ggml_view_3d(ctx, locations_xy_last, per_head, config.heads, queries,
                     per_head * sizeof(float),
                     per_head * config.heads * sizeof(float), component_bytes);

    std::vector<float> head_data(static_cast<size_t>(samples));
    for (int64_t query = 0; query < queries; ++query) {
        for (int head = 0; head < config.heads; ++head) {
            for (int point = 0; point < config.points; ++point) {
                head_data[static_cast<size_t>(
                    point + config.points * (head + config.heads * query))] =
                    static_cast<float>(head);
            }
        }
    }
    ggml_tensor *head_ids = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, config.points, config.heads, queries);
    if (!session.set_input(head_ids, head_data.data(), ggml_nbytes(head_ids),
                           error))
        return false;

    ggml_tensor *accumulated = nullptr;
    int64_t token_start = 0;
    for (int level = 0; level < config.levels; ++level) {
        const float width = static_cast<float>(config.widths[level]);
        const float height = static_cast<float>(config.heights[level]);
        ggml_tensor *x = ggml_scale_bias(
            ctx, level_slice(ctx, x_all, level, config.points), width, -0.5F);
        ggml_tensor *y = ggml_scale_bias(
            ctx, level_slice(ctx, y_all, level, config.points), height, -0.5F);
        ggml_tensor *x0 = ggml_floor(ctx, x);
        ggml_tensor *y0 = ggml_floor(ctx, y);
        ggml_tensor *x1 = ggml_scale_bias(ctx, x0, 1.0F, 1.0F);
        ggml_tensor *y1 = ggml_scale_bias(ctx, y0, 1.0F, 1.0F);
        ggml_tensor *dx = checked_sub(ctx, x, x0);
        ggml_tensor *dy = checked_sub(ctx, y, y0);
        ggml_tensor *one_minus_dx = ggml_scale_bias(ctx, dx, -1.0F, 1.0F);
        ggml_tensor *one_minus_dy = ggml_scale_bias(ctx, dy, -1.0F, 1.0F);
        ggml_tensor *level_weights =
            level_slice(ctx, weights, level, config.points);

        auto sample_corner = [&](ggml_tensor *xi, ggml_tensor *yi,
                                 ggml_tensor *wx, ggml_tensor *wy) {
            ggml_tensor *valid =
                checked_mul(ctx, coordinate_validity(ctx, xi, width),
                            coordinate_validity(ctx, yi, height));
            ggml_tensor *xc = ggml_clamp(ctx, xi, 0.0F, width - 1.0F);
            ggml_tensor *yc = ggml_clamp(ctx, yi, 0.0F, height - 1.0F);
            ggml_tensor *token =
                checked_add(ctx,
                            ggml_scale_bias(ctx, yc, width,
                                            static_cast<float>(token_start)),
                            xc);
            ggml_tensor *row = checked_add(
                ctx, ggml_scale(ctx, token, static_cast<float>(config.heads)),
                head_ids);
            ggml_tensor *indices = ggml_reshape_1d(
                ctx, ggml_cast(ctx, row, GGML_TYPE_I32), samples);
            ggml_tensor *gathered = ggml_reshape_4d(
                ctx, ggml_get_rows(ctx, value_rows, indices), config.head_dim,
                config.points, config.heads, queries);
            ggml_tensor *coefficient =
                checked_mul(ctx, level_weights,
                            checked_mul(ctx, valid, checked_mul(ctx, wx, wy)));
            coefficient = ggml_reshape_4d(ctx, coefficient, 1, config.points,
                                          config.heads, queries);
            coefficient = ggml_repeat_4d(ctx, coefficient, config.head_dim,
                                         config.points, config.heads, queries);
            return checked_mul(ctx, gathered, coefficient);
        };

        ggml_tensor *sampled = checked_add(
            ctx,
            checked_add(ctx, sample_corner(x0, y0, one_minus_dx, one_minus_dy),
                        sample_corner(x1, y0, dx, one_minus_dy)),
            checked_add(ctx, sample_corner(x0, y1, one_minus_dx, dy),
                        sample_corner(x1, y1, dx, dy)));
        sampled = ggml_cont(ctx, ggml_permute(ctx, sampled, 1, 0, 2, 3));
        sampled = ggml_reshape_3d(ctx, ggml_sum_rows(ctx, sampled),
                                  config.head_dim, config.heads, queries);
        accumulated =
            accumulated ? checked_add(ctx, accumulated, sampled) : sampled;
        token_start +=
            static_cast<int64_t>(config.heights[level]) * config.widths[level];
    }

    ggml_tensor *result = ggml_reshape_2d(
        ctx, accumulated, static_cast<int64_t>(config.head_dim) * config.heads,
        queries);
    // Preserve the anchor dependency expected by existing callers and tests.
    result = checked_add(ctx, result, ggml_scale(ctx, anchor, 0.0F));
    *output = result;
    return session.checkpoint({"msda.sampler.output", "msda", "Q,H"}, *output,
                              error);
}

} // namespace wevision
