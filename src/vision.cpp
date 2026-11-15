#include "vision.hpp"

#include "runtime.hpp"

#include "graph_ops.hpp"
#include "parallel.hpp"

#include "ggml.h"

#include <cmath>
#include <limits>
#include <utility>

namespace wevision {
namespace {

bool valid_image(const RgbImage &image) {
    return image.width > 0 && image.height > 0 &&
           image.pixels.size() ==
               static_cast<size_t>(image.width) * image.height * 3U;
}

int round_to_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5F)
        return static_cast<int>(lower);
    if (fraction > 0.5F)
        return static_cast<int>(lower + 1.0F);
    const int integer = static_cast<int>(lower);
    return (integer & 1) == 0 ? integer : integer + 1;
}

bool valid_config(const VisionPreprocessConfig &config) {
    return config.patch_size > 0 && config.temporal_patch_size > 0 &&
           config.spatial_merge_size > 0 && config.min_pixels > 0 &&
           config.max_pixels >= config.min_pixels &&
           config.image_std[0] != 0.0F && config.image_std[1] != 0.0F &&
           config.image_std[2] != 0.0F;
}

float cubic_weight(float distance) {
    // Keys cubic with a=-0.5, the kernel used by Pillow's BICUBIC mode.
    constexpr float a = -0.5F;
    distance = std::fabs(distance);
    if (distance <= 1.0F) {
        return (a + 2.0F) * distance * distance * distance -
               (a + 3.0F) * distance * distance + 1.0F;
    }
    if (distance < 2.0F) {
        return a * distance * distance * distance -
               5.0F * a * distance * distance + 8.0F * a * distance - 4.0F * a;
    }
    return 0.0F;
}

bool resize_rgb_bicubic(const RgbImage &source, int target_width,
                        int target_height, RgbImage *output) {
    if (output == nullptr || !valid_image(source) || target_width <= 0 ||
        target_height <= 0)
        return false;
    RgbImage result;
    result.width = target_width;
    result.height = target_height;
    result.pixels.resize(static_cast<size_t>(target_width) * target_height *
                         3U);
    const float scale_x = static_cast<float>(source.width) / target_width;
    const float scale_y = static_cast<float>(source.height) / target_height;
    // Pillow enables antialiasing for reducing resizes: widen the cubic kernel
    // in source coordinates and normalize its weights at image boundaries.
    const float filter_x = scale_x > 1.0F ? scale_x : 1.0F;
    const float filter_y = scale_y > 1.0F ? scale_y : 1.0F;
    const int radius_x = static_cast<int>(std::ceil(2.0F * filter_x));
    const int radius_y = static_cast<int>(std::ceil(2.0F * filter_y));
    const int tap_x = 2 * radius_x;
    const int tap_y = 2 * radius_y;
    std::vector<int> x_indices(static_cast<size_t>(target_width) * tap_x);
    std::vector<float> x_weights(static_cast<size_t>(target_width) * tap_x);
    for (int target_x = 0; target_x < target_width; ++target_x) {
        const float source_x =
            (static_cast<float>(target_x) + 0.5F) * scale_x - 0.5F;
        const int center_x = static_cast<int>(std::floor(source_x));
        float weight_sum = 0.0F;
        for (int k = 0; k < tap_x; ++k) {
            const int x = center_x - radius_x + 1 + k;
            const int clamped_x =
                x < 0 ? 0 : (x >= source.width ? source.width - 1 : x);
            const float w =
                cubic_weight((source_x - static_cast<float>(x)) / filter_x) /
                filter_x;
            x_indices[static_cast<size_t>(target_x) * tap_x + k] = clamped_x;
            x_weights[static_cast<size_t>(target_x) * tap_x + k] = w;
            weight_sum += w;
        }
        if (weight_sum != 0.0F) {
            const float inv = 1.0F / weight_sum;
            for (int k = 0; k < tap_x; ++k)
                x_weights[static_cast<size_t>(target_x) * tap_x + k] *= inv;
        }
    }

    std::vector<float> horiz(static_cast<size_t>(source.height) *
                             target_width * 3U);
    parallel_for(static_cast<size_t>(source.height), [&](size_t row) {
        const uint8_t *src_row =
            source.pixels.data() + row * static_cast<size_t>(source.width) * 3U;
        float *dst_row =
            horiz.data() + row * static_cast<size_t>(target_width) * 3U;
        for (int target_x = 0; target_x < target_width; ++target_x) {
            const int *idx =
                x_indices.data() + static_cast<size_t>(target_x) * tap_x;
            const float *w =
                x_weights.data() + static_cast<size_t>(target_x) * tap_x;
            float sum0 = 0.0F;
            float sum1 = 0.0F;
            float sum2 = 0.0F;
            for (int k = 0; k < tap_x; ++k) {
                const uint8_t *px = src_row + static_cast<size_t>(idx[k]) * 3U;
                const float wk = w[k];
                sum0 += wk * px[0];
                sum1 += wk * px[1];
                sum2 += wk * px[2];
            }
            float *out_px = dst_row + static_cast<size_t>(target_x) * 3U;
            out_px[0] = sum0;
            out_px[1] = sum1;
            out_px[2] = sum2;
        }
    });

    std::vector<int> y_indices(static_cast<size_t>(target_height) * tap_y);
    std::vector<float> y_weights(static_cast<size_t>(target_height) * tap_y);
    for (int target_y = 0; target_y < target_height; ++target_y) {
        const float source_y =
            (static_cast<float>(target_y) + 0.5F) * scale_y - 0.5F;
        const int center_y = static_cast<int>(std::floor(source_y));
        float weight_sum = 0.0F;
        for (int k = 0; k < tap_y; ++k) {
            const int y = center_y - radius_y + 1 + k;
            const int clamped_y =
                y < 0 ? 0 : (y >= source.height ? source.height - 1 : y);
            const float w =
                cubic_weight((source_y - static_cast<float>(y)) / filter_y) /
                filter_y;
            y_indices[static_cast<size_t>(target_y) * tap_y + k] = clamped_y;
            y_weights[static_cast<size_t>(target_y) * tap_y + k] = w;
            weight_sum += w;
        }
        if (weight_sum != 0.0F) {
            const float inv = 1.0F / weight_sum;
            for (int k = 0; k < tap_y; ++k)
                y_weights[static_cast<size_t>(target_y) * tap_y + k] *= inv;
        }
    }

    const size_t row_floats = static_cast<size_t>(target_width) * 3U;
    parallel_for(static_cast<size_t>(target_height), [&](size_t row) {
        const int target_y = static_cast<int>(row);
        const int *idx =
            y_indices.data() + static_cast<size_t>(target_y) * tap_y;
        const float *w =
            y_weights.data() + static_cast<size_t>(target_y) * tap_y;
        uint8_t *dst_row = result.pixels.data() + row * row_floats;
        for (size_t i = 0; i < row_floats; ++i) {
            float sum = 0.0F;
            for (int k = 0; k < tap_y; ++k) {
                sum += w[k] *
                       horiz[static_cast<size_t>(idx[k]) * row_floats + i];
            }
            const int rounded = static_cast<int>(std::lround(sum));
            dst_row[i] = static_cast<uint8_t>(
                rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
        }
    });
    *output = std::move(result);
    return true;
}

} // namespace

bool smart_resize(int source_height, int source_width,
                  const VisionPreprocessConfig &config,
                  SmartResizeShape *output) {
    if (output == nullptr || source_height <= 0 || source_width <= 0 ||
        !valid_config(config)) {
        return false;
    }
    const int factor = config.patch_size * config.spatial_merge_size;
    if (factor <= 0 ||
        source_height > std::numeric_limits<int>::max() / source_width) {
        return false;
    }
    const float shortest = static_cast<float>(
        source_height < source_width ? source_height : source_width);
    const float longest = static_cast<float>(
        source_height > source_width ? source_height : source_width);
    if (longest / shortest > 200.0F)
        return false;

    int height =
        round_to_even(static_cast<float>(source_height) / factor) * factor;
    int width =
        round_to_even(static_cast<float>(source_width) / factor) * factor;
    if (height <= 0 || width <= 0)
        return false;
    const int64_t pixels = static_cast<int64_t>(height) * width;
    if (pixels > config.max_pixels) {
        const float beta = std::sqrt(static_cast<float>(source_height) *
                                     static_cast<float>(source_width) /
                                     static_cast<float>(config.max_pixels));
        height = static_cast<int>(std::floor(source_height / beta / factor)) *
                 factor;
        width =
            static_cast<int>(std::floor(source_width / beta / factor)) * factor;
        height = height < factor ? factor : height;
        width = width < factor ? factor : width;
    } else if (pixels < config.min_pixels) {
        const float beta = std::sqrt(static_cast<float>(config.min_pixels) /
                                     (static_cast<float>(source_height) *
                                      static_cast<float>(source_width)));
        height =
            static_cast<int>(std::ceil(source_height * beta / factor)) * factor;
        width =
            static_cast<int>(std::ceil(source_width * beta / factor)) * factor;
    }
    output->height = height;
    output->width = width;
    return true;
}

bool patchify_resized_rgb(const RgbImage &image,
                          const VisionPreprocessConfig &config,
                          PatchifiedImage *output) {
    if (output == nullptr || !valid_config(config) || !valid_image(image)) {
        return false;
    }
    const int factor = config.patch_size * config.spatial_merge_size;
    if (image.width % factor != 0 || image.height % factor != 0)
        return false;
    const int grid_h = image.height / config.patch_size;
    const int grid_w = image.width / config.patch_size;
    const int patch_dim =
        3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    const int patch_count = grid_h * grid_w;
    if (patch_count <= 0 || patch_dim <= 0)
        return false;

    PatchifiedImage result;
    result.grid_h = grid_h;
    result.grid_w = grid_w;
    result.patch_dim = patch_dim;
    result.values.resize(static_cast<size_t>(patch_count) * patch_dim);

    const int merge = config.spatial_merge_size;
    const int blocks_x = grid_w / merge;
    const int blocks_y = grid_h / merge;
    const size_t patches_per_block = static_cast<size_t>(merge) * merge;
    parallel_for(static_cast<size_t>(blocks_x) * blocks_y, [&](size_t block) {
        const int block_y = static_cast<int>(block) / blocks_x;
        const int block_x = static_cast<int>(block) % blocks_x;
        size_t destination = block * patches_per_block * patch_dim;
        for (int intra_y = 0; intra_y < merge; ++intra_y) {
            for (int intra_x = 0; intra_x < merge; ++intra_x) {
                const int patch_y =
                    (block_y * merge + intra_y) * config.patch_size;
                const int patch_x =
                    (block_x * merge + intra_x) * config.patch_size;
                for (int channel = 0; channel < 3; ++channel) {
                    for (int temporal = 0;
                         temporal < config.temporal_patch_size; ++temporal) {
                        (void)temporal; // A still image is expanded across
                                        // temporal patches.
                        for (int y = 0; y < config.patch_size; ++y) {
                            for (int x = 0; x < config.patch_size; ++x) {
                                const size_t source =
                                    (static_cast<size_t>(patch_y + y) *
                                         image.width +
                                     patch_x + x) *
                                        3U +
                                    static_cast<size_t>(channel);
                                const float rescaled =
                                    static_cast<float>(image.pixels[source]) /
                                    255.0F;
                                result.values[destination++] =
                                    (rescaled - config.image_mean[channel]) /
                                    config.image_std[channel];
                            }
                        }
                    }
                }
            }
        }
    });
    *output = std::move(result);
    return true;
}

bool prepare_vision_image(const RgbImage &source,
                          const VisionPreprocessConfig &config,
                          VisionImageInput *output) {
    if (output == nullptr || !valid_image(source))
        return false;
    SmartResizeShape target;
    if (!smart_resize(source.height, source.width, config, &target))
        return false;
    RgbImage resized;
    if (!resize_rgb_bicubic(source, target.width, target.height, &resized))
        return false;
    VisionImageInput result;
    if (!patchify_resized_rgb(resized, config, &result.patches))
        return false;
    result.grid_thw[0] = 1;
    result.grid_thw[1] = result.patches.grid_h;
    result.grid_thw[2] = result.patches.grid_w;
    *output = std::move(result);
    return true;
}

bool prepare_mask_image(const RgbImage &source, int cnn_image_size,
                        MaskImageInput *output) {
    if (output == nullptr || !valid_image(source) || cnn_image_size <= 0)
        return false;
    RgbImage resized;
    if (!resize_rgb_bicubic(source, cnn_image_size, cnn_image_size, &resized))
        return false;
    MaskImageInput result;
    result.original_width = source.width;
    result.original_height = source.height;
    result.width = cnn_image_size;
    result.height = cnn_image_size;
    const size_t plane = static_cast<size_t>(cnn_image_size) * cnn_image_size;
    result.rgb.resize(3U * plane);
    for (int y = 0; y < cnn_image_size; ++y) {
        for (int x = 0; x < cnn_image_size; ++x) {
            const size_t input =
                (static_cast<size_t>(y) * cnn_image_size + x) * 3U;
            const size_t pixel = static_cast<size_t>(y) * cnn_image_size + x;
            for (int channel = 0; channel < 3; ++channel) {
                result.rgb[static_cast<size_t>(channel) * plane + pixel] =
                    static_cast<float>(resized.pixels[input + channel]) /
                    255.0F;
            }
        }
    }
    *output = std::move(result);
    return true;
}

bool build_vision_position_data(int grid_h, int grid_w, int spatial_merge_size,
                                int position_table_side,
                                VisionPositionData *output) {
    if (output == nullptr || grid_h <= 0 || grid_w <= 0 ||
        spatial_merge_size <= 0 || position_table_side <= 0 ||
        grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0) {
        return false;
    }
    const int token_count = grid_h * grid_w;
    VisionPositionData result;
    result.token_count = token_count;
    result.axial_hw.reserve(static_cast<size_t>(token_count) * 2U);
    result.interpolation_ids.reserve(static_cast<size_t>(token_count) * 4U);
    result.interpolation_weights.reserve(static_cast<size_t>(token_count) * 4U);
    const int merge = spatial_merge_size;
    const int blocks_w = grid_w / merge;
    for (int token = 0; token < token_count; ++token) {
        const int in_col = token % merge;
        const int in_row = (token / merge) % merge;
        const int block_col = (token / (merge * merge)) % blocks_w;
        const int block_row = token / (merge * merge * blocks_w);
        const int row = block_row * merge + in_row;
        const int col = block_col * merge + in_col;
        result.axial_hw.push_back(row);
        result.axial_hw.push_back(col);

        const float src_row = grid_h == 1 ? 0.0F
                                          : static_cast<float>(row) *
                                                (position_table_side - 1) /
                                                (grid_h - 1);
        const float src_col = grid_w == 1 ? 0.0F
                                          : static_cast<float>(col) *
                                                (position_table_side - 1) /
                                                (grid_w - 1);
        const int row0_unclamped = static_cast<int>(std::floor(src_row));
        const int col0_unclamped = static_cast<int>(std::floor(src_col));
        const int row0 = row0_unclamped < 0
                             ? 0
                             : (row0_unclamped >= position_table_side
                                    ? position_table_side - 1
                                    : row0_unclamped);
        const int col0 = col0_unclamped < 0
                             ? 0
                             : (col0_unclamped >= position_table_side
                                    ? position_table_side - 1
                                    : col0_unclamped);
        const int row1 = row0 + 1 < position_table_side ? row0 + 1 : row0;
        const int col1 = col0 + 1 < position_table_side ? col0 + 1 : col0;
        const float fy = src_row - std::floor(src_row);
        const float fx = src_col - std::floor(src_col);
        result.interpolation_ids.insert(result.interpolation_ids.end(),
                                        {row0 * position_table_side + col0,
                                         row0 * position_table_side + col1,
                                         row1 * position_table_side + col0,
                                         row1 * position_table_side + col1});
        result.interpolation_weights.insert(result.interpolation_weights.end(),
                                            {(1.0F - fy) * (1.0F - fx),
                                             (1.0F - fy) * fx, fy * (1.0F - fx),
                                             fy * fx});
    }
    *output = std::move(result);
    return true;
}

bool build_vision_rotary_data(const VisionPositionData &positions, int head_dim,
                              float rope_theta, VisionRotaryData *output) {
    if (output == nullptr || positions.token_count <= 0 || head_dim <= 0 ||
        head_dim % 4 != 0 || rope_theta <= 1.0F ||
        positions.axial_hw.size() !=
            static_cast<size_t>(positions.token_count) * 2U) {
        return false;
    }
    const int frequency_count = head_dim / 4;
    VisionRotaryData result;
    result.token_count = positions.token_count;
    result.head_dim = head_dim;
    result.cos.resize(static_cast<size_t>(positions.token_count) * head_dim);
    result.sin.resize(static_cast<size_t>(positions.token_count) * head_dim);
    for (int token = 0; token < positions.token_count; ++token) {
        const float row = static_cast<float>(
            positions.axial_hw[static_cast<size_t>(token) * 2U]);
        const float col = static_cast<float>(
            positions.axial_hw[static_cast<size_t>(token) * 2U + 1U]);
        for (int index = 0; index < frequency_count; ++index) {
            const float inv_freq =
                1.0F / std::pow(rope_theta,
                                static_cast<float>(2 * index) / (head_dim / 2));
            const float row_angle = row * inv_freq;
            const float col_angle = col * inv_freq;
            const size_t base = static_cast<size_t>(token) * head_dim;
            // Python: cat([h, w]) then repeats that half once.
            result.cos[base + index] = std::cos(row_angle);
            result.sin[base + index] = std::sin(row_angle);
            result.cos[base + frequency_count + index] = std::cos(col_angle);
            result.sin[base + frequency_count + index] = std::sin(col_angle);
            result.cos[base + 2 * frequency_count + index] =
                std::cos(row_angle);
            result.sin[base + 2 * frequency_count + index] =
                std::sin(row_angle);
            result.cos[base + 3 * frequency_count + index] =
                std::cos(col_angle);
            result.sin[base + 3 * frequency_count + index] =
                std::sin(col_angle);
        }
    }
    *output = std::move(result);
    return true;
}

bool build_vision_patch_embed(GraphSession &session,
                              const PatchifiedImage &input,
                              ggml_tensor **output, std::string *error) {
    if (output == nullptr || input.patch_dim <= 0 || input.values.empty() ||
        input.values.size() % static_cast<size_t>(input.patch_dim) != 0U) {
        set_error(error, "invalid packed vision patches");
        return false;
    }
    const int64_t hidden =
        static_cast<int64_t>(session.model().config.vision.hidden_size);
    ggml_context *context = session.context();
    ggml_tensor *weight = session.weight("wv.vision.patch_embed.weight", error);
    ggml_tensor *bias = session.weight("wv.vision.patch_embed.bias", error);
    if (weight == nullptr || bias == nullptr)
        return false;
    if (ggml_nelements(weight) != input.patch_dim * hidden ||
        ggml_nelements(bias) != hidden) {
        set_error(error, "invalid patch embedding weight shape");
        return false;
    }
    const int64_t token_count =
        static_cast<int64_t>(input.values.size() / input.patch_dim);
    ggml_tensor *patches = ggml_new_tensor_2d(context, GGML_TYPE_F32,
                                              input.patch_dim, token_count);
    // GGUF retains Conv3D's [out,C,T,H,W] dimensions. It is contiguous per
    // output channel, so reshape exposes the equivalent [out, packed_patch]
    // projection without copying or changing the stored weight layout.
    ggml_tensor *matrix =
        ggml_reshape_2d(context, weight, input.patch_dim, hidden);
    ggml_tensor *projected = checked_mul_mat(context, matrix, patches);
    ggml_tensor *bias_f32 = ggml_cast(context, bias, GGML_TYPE_F32);
    ggml_tensor *result = checked_add(
        context, projected, ggml_reshape_2d(context, bias_f32, hidden, 1));
    if (!session.set_input(patches, input.values.data(), ggml_nbytes(patches),
                           error))
        return false;
    *output = result;
    if (!session.checkpoint(
            {"vision.patch_embed.output", "vision", "patch,hidden"}, result,
            error))
        return false;
    return true;
}

bool build_vision_positioned_input(GraphSession &session,
                                   ggml_tensor *patch_embedding,
                                   const VisionPositionData &positions,
                                   ggml_tensor **output, std::string *error) {
    if (output == nullptr || patch_embedding == nullptr ||
        patch_embedding->type != GGML_TYPE_F32 ||
        patch_embedding->ne[0] !=
            static_cast<int64_t>(session.model().config.vision.hidden_size) ||
        patch_embedding->ne[1] != positions.token_count ||
        positions.interpolation_ids.size() !=
            static_cast<size_t>(positions.token_count) * 4U ||
        positions.interpolation_weights.size() !=
            static_cast<size_t>(positions.token_count) * 4U) {
        set_error(error, "invalid vision position input");
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *table = session.weight("wv.vision.pos_embd.weight", error);
    if (table == nullptr)
        return false;
    if (table->ne[0] !=
            static_cast<int64_t>(session.model().config.vision.hidden_size) ||
        table->ne[1] !=
            static_cast<int64_t>(
                session.model().config.vision.position_table_size)) {
        set_error(error, "invalid vision position table shape");
        return false;
    }
    ggml_tensor *position_sum = nullptr;
    for (int tap = 0; tap < 4; ++tap) {
        ggml_tensor *indices =
            ggml_new_tensor_1d(context, GGML_TYPE_I32, positions.token_count);
        ggml_tensor *weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, 1,
                                                  positions.token_count);
        std::vector<int32_t> ids(static_cast<size_t>(positions.token_count));
        std::vector<float> values(static_cast<size_t>(positions.token_count));
        for (int token = 0; token < positions.token_count; ++token) {
            ids[static_cast<size_t>(token)] =
                positions
                    .interpolation_ids[static_cast<size_t>(token) * 4U + tap];
            values[static_cast<size_t>(token)] =
                positions
                    .interpolation_weights[static_cast<size_t>(token) * 4U +
                                           tap];
        }
        if (!session.set_input(indices, ids.data(), ggml_nbytes(indices),
                               error) ||
            !session.set_input(weights, values.data(), ggml_nbytes(weights),
                               error))
            return false;
        ggml_tensor *gathered = ggml_cast(
            context, ggml_get_rows(context, table, indices), GGML_TYPE_F32);
        ggml_tensor *weighted = checked_mul(context, gathered, weights);
        position_sum = position_sum == nullptr
                           ? weighted
                           : checked_add(context, position_sum, weighted);
    }
    *output = checked_add(context, patch_embedding, position_sum);
    if (!session.checkpoint(
            {"vision.positioned_input", "vision", "patch,hidden"}, *output,
            error))
        return false;
    return true;
}

bool build_vision_layer_norm(GraphSession &session, ggml_tensor *input,
                             const std::string &weight_name,
                             const std::string &bias_name, ggml_tensor **output,
                             std::string *error) {
    if (output == nullptr || input == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] <= 0) {
        set_error(error, "invalid vision layer norm input");
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *weight = session.weight(weight_name, error);
    ggml_tensor *bias = session.weight(bias_name, error);
    if (weight == nullptr || bias == nullptr)
        return false;
    if (ggml_nelements(weight) != input->ne[0] ||
        ggml_nelements(bias) != input->ne[0]) {
        set_error(error,
                  "invalid vision layer norm parameter shape: " + weight_name);
        return false;
    }
    // ggml_norm implements mean-centred LayerNorm; ggml_rms_norm is the
    // separate RMSNorm op used by the text decoder.
    ggml_tensor *normalized =
        ggml_norm(context, input, session.model().config.vision.layer_norm_eps);
    ggml_tensor *scale = ggml_reshape_2d(
        context, ggml_cast(context, weight, GGML_TYPE_F32), input->ne[0], 1);
    ggml_tensor *shift = ggml_reshape_2d(
        context, ggml_cast(context, bias, GGML_TYPE_F32), input->ne[0], 1);
    *output =
        checked_add(context, checked_mul(context, normalized, scale), shift);
    return true;
}

bool build_vision_linear(GraphSession &session, ggml_tensor *input,
                         const std::string &weight_name,
                         const std::string &bias_name, int output_size,
                         ggml_tensor **output, std::string *error) {
    if (output == nullptr || input == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] <= 0 || output_size <= 0) {
        set_error(error, "invalid vision linear input");
        return false;
    }
    ggml_context *context = session.context();
    ggml_tensor *weight = session.weight(weight_name, error);
    ggml_tensor *bias = session.weight(bias_name, error);
    if (weight == nullptr || bias == nullptr)
        return false;
    if (ggml_nelements(weight) != input->ne[0] * output_size ||
        ggml_nelements(bias) != output_size) {
        set_error(error,
                  "invalid vision linear parameter shape: " + weight_name);
        return false;
    }
    ggml_tensor *matrix =
        ggml_reshape_2d(context, weight, input->ne[0], output_size);
    ggml_tensor *projected = checked_mul_mat(context, matrix, input);
    ggml_tensor *bias_f32 = ggml_cast(context, bias, GGML_TYPE_F32);
    *output = checked_add(context, projected,
                          ggml_reshape_2d(context, bias_f32, output_size, 1));
    return true;
}

bool build_vision_qkv_rotary(GraphSession &session, ggml_tensor *packed_qkv,
                             const VisionRotaryData &rotary,
                             VisionAttentionQKV *output, std::string *error) {
    const int64_t hidden =
        static_cast<int64_t>(session.model().config.vision.hidden_size);
    const int64_t heads =
        static_cast<int64_t>(session.model().config.vision.head_count);
    const int64_t head_dim = hidden / heads;
    if (output == nullptr || packed_qkv == nullptr ||
        packed_qkv->type != GGML_TYPE_F32 || packed_qkv->ne[0] != 3 * hidden ||
        packed_qkv->ne[1] != rotary.token_count ||
        rotary.head_dim != head_dim ||
        rotary.cos.size() != static_cast<size_t>(rotary.token_count) *
                                 static_cast<size_t>(head_dim) ||
        rotary.sin.size() != rotary.cos.size()) {
        set_error(error, "invalid vision QKV rotary input");
        return false;
    }
    ggml_context *context = session.context();
    const int64_t tokens = rotary.token_count;
    ggml_tensor *cos =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, head_dim, tokens);
    ggml_tensor *sin =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, head_dim, tokens);
    std::vector<float> cos_transposed(rotary.cos.size()),
        sin_transposed(rotary.sin.size());
    for (int token = 0; token < rotary.token_count; ++token)
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            cos_transposed[static_cast<size_t>(dim) +
                           static_cast<size_t>(head_dim) * token] =
                rotary.cos[static_cast<size_t>(token) *
                               static_cast<size_t>(head_dim) +
                           dim];
            sin_transposed[static_cast<size_t>(dim) +
                           static_cast<size_t>(head_dim) * token] =
                rotary.sin[static_cast<size_t>(token) *
                               static_cast<size_t>(head_dim) +
                           dim];
        }
    if (!session.set_input(cos, cos_transposed.data(), ggml_nbytes(cos),
                           error) ||
        !session.set_input(sin, sin_transposed.data(), ggml_nbytes(sin), error))
        return false;
    auto view = [&](size_t offset) {
        ggml_tensor *slice = ggml_view_2d(context, packed_qkv, hidden, tokens,
                                          packed_qkv->nb[1], offset);
        // The packed [Q,K,V] projection has a wider row stride. Materialize
        // only this request activation before changing it to head-major view.
        return ggml_reshape_3d(context, ggml_cont(context, slice), head_dim,
                               heads, tokens);
    };
    auto rotate = [&](ggml_tensor *x) {
        ggml_tensor *first = ggml_view_3d(context, x, head_dim / 2, heads,
                                          tokens, x->nb[1], x->nb[2], 0);
        ggml_tensor *second = ggml_view_3d(
            context, x, head_dim / 2, heads, tokens, x->nb[1], x->nb[2],
            static_cast<size_t>(head_dim) / 2 * sizeof(float));
        return checked_concat(context, ggml_neg(context, second), first, 0);
    };
    auto apply = [&](ggml_tensor *x) {
        ggml_tensor *c = ggml_reshape_3d(context, cos, head_dim, 1, tokens);
        ggml_tensor *s = ggml_reshape_3d(context, sin, head_dim, 1, tokens);
        return checked_add(context, checked_mul(context, x, c),
                           checked_mul(context, rotate(x), s));
    };
    output->query = apply(view(0));
    output->key = apply(view(static_cast<size_t>(hidden) * sizeof(float)));
    output->value = view(2 * static_cast<size_t>(hidden) * sizeof(float));
    return true;
}

bool build_vision_attention(GraphSession &session,
                            const VisionAttentionQKV &qkv, ggml_tensor **output,
                            std::string *error) {
    const int64_t heads =
        static_cast<int64_t>(session.model().config.vision.head_count);
    const int64_t head_dim =
        static_cast<int64_t>(session.model().config.vision.hidden_size / heads);
    if (output == nullptr || qkv.query == nullptr || qkv.key == nullptr ||
        qkv.value == nullptr || qkv.query->ne[0] != head_dim ||
        qkv.query->ne[1] != heads || qkv.key->ne[0] != head_dim ||
        qkv.value->ne[0] != head_dim || qkv.key->ne[1] != heads ||
        qkv.value->ne[1] != heads || qkv.query->ne[2] != qkv.key->ne[2] ||
        qkv.query->ne[2] != qkv.value->ne[2]) {
        set_error(error, "invalid vision attention tensors");
        return false;
    }
    ggml_context *context = session.context();
    // [D,H,T] -> [D,T,H]: one independent attention matrix per head.
    ggml_tensor *query = ggml_permute(context, qkv.query, 0, 2, 1, 3);
    ggml_tensor *key = ggml_permute(context, qkv.key, 0, 2, 1, 3);
    ggml_tensor *value = ggml_permute(context, qkv.value, 0, 2, 1, 3);
    // Native fused attention keeps CPU and Metal on the same non-causal
    // implementation. Its result is already [D,H,T].
    ggml_tensor *head_major = ggml_flash_attn_ext(
        context, query, key, value, nullptr,
        1.0F / std::sqrt(static_cast<float>(head_dim)), 0.0F, 0.0F);
    *output = head_major;
    return true;
}

bool build_vision_block(GraphSession &session, int layer, ggml_tensor *input,
                        const VisionRotaryData &rotary, ggml_tensor **output,
                        std::string *error) {
    const auto &config = session.model().config.vision;
    const int64_t hidden = static_cast<int64_t>(config.hidden_size);
    const int64_t intermediate = static_cast<int64_t>(config.intermediate_size);
    if (layer < 0 || output == nullptr || input == nullptr ||
        input->ne[0] != hidden) {
        set_error(error, "invalid vision block input");
        return false;
    }
    const std::string prefix = "wv.vision.blk." + std::to_string(layer) + ".";
    ggml_tensor *norm1 = nullptr;
    ggml_tensor *qkv = nullptr;
    VisionAttentionQKV heads;
    ggml_tensor *attended = nullptr;
    ggml_tensor *attn_flat = nullptr;
    ggml_tensor *attn_out = nullptr;
    ggml_tensor *norm2 = nullptr;
    ggml_tensor *up = nullptr;
    ggml_tensor *down = nullptr;
    if (!build_vision_layer_norm(session, input, prefix + "attn_norm.weight",
                                 prefix + "attn_norm.bias", &norm1, error) ||
        !build_vision_linear(session, norm1, prefix + "attn_qkv.weight",
                             prefix + "attn_qkv.bias",
                             static_cast<int>(3 * hidden), &qkv, error) ||
        !build_vision_qkv_rotary(session, qkv, rotary, &heads, error) ||
        !build_vision_attention(session, heads, &attended, error))
        return false;
    attn_flat =
        ggml_reshape_2d(session.context(), attended, hidden, input->ne[1]);
    if (!build_vision_linear(session, attn_flat, prefix + "attn_output.weight",
                             prefix + "attn_output.bias",
                             static_cast<int>(hidden), &attn_out, error))
        return false;
    ggml_tensor *after_attention =
        checked_add(session.context(), input, attn_out);
    if (!build_vision_layer_norm(session, after_attention,
                                 prefix + "ffn_norm.weight",
                                 prefix + "ffn_norm.bias", &norm2, error) ||
        !build_vision_linear(session, norm2, prefix + "ffn_up.weight",
                             prefix + "ffn_up.bias",
                             static_cast<int>(intermediate), &up, error) ||
        !build_vision_linear(session, ggml_gelu(session.context(), up),
                             prefix + "ffn_down.weight",
                             prefix + "ffn_down.bias", static_cast<int>(hidden),
                             &down, error))
        return false;
    *output = checked_add(session.context(), after_attention, down);
    const std::string index =
        layer < 10 ? "0" + std::to_string(layer) : std::to_string(layer);
    if (!session.checkpoint(
            {"vision.block." + index + ".output", "vision", "patch,hidden"},
            *output, error))
        return false;
    return true;
}

bool build_vision_merger(GraphSession &session, ggml_tensor *input,
                         const std::string &prefix, bool norm_after_shuffle,
                         ggml_tensor **output, std::string *error) {
    const auto &vision = session.model().config.vision;
    const int64_t merge = static_cast<int64_t>(vision.spatial_merge_size);
    const int64_t merged_width =
        merge * merge * static_cast<int64_t>(vision.hidden_size);
    const int64_t text_hidden =
        static_cast<int64_t>(session.model().config.text.hidden_size);
    if (output == nullptr || input == nullptr || input->type != GGML_TYPE_F32 ||
        input->ne[0] != static_cast<int64_t>(vision.hidden_size) ||
        input->ne[1] % (merge * merge) != 0) {
        set_error(error, "invalid vision merger input");
        return false;
    }
    ggml_tensor *value = input;
    if (!norm_after_shuffle) {
        if (!build_vision_layer_norm(session, value, prefix + ".norm.weight",
                                     prefix + ".norm.bias", &value, error))
            return false;
    }
    value = ggml_reshape_2d(session.context(), value, merged_width,
                            value->ne[1] / (merge * merge));
    if (norm_after_shuffle &&
        !build_vision_layer_norm(session, value, prefix + ".norm.weight",
                                 prefix + ".norm.bias", &value, error))
        return false;
    ggml_tensor *up = nullptr;
    ggml_tensor *down = nullptr;
    if (!build_vision_linear(
            session, value, prefix + ".linear_fc1.weight",
            prefix + ".linear_fc1.bias",
            static_cast<int>(static_cast<int64_t>(vision.intermediate_size)),
            &up, error) ||
        !build_vision_linear(session, ggml_gelu(session.context(), up),
                             prefix + ".linear_fc2.weight",
                             prefix + ".linear_fc2.bias",
                             static_cast<int>(text_hidden), &down, error))
        return false;
    *output = down;
    std::string name;
    if (prefix == "wv.vision.merger")
        name = "vision.merger.output";
    else if (prefix.rfind("wv.vision.deepstack.", 0) == 0)
        name = "vision.deepstack." +
               prefix.substr(std::string("wv.vision.deepstack.").size()) +
               ".output";
    if (!name.empty() &&
        !session.checkpoint({name, "vision", "merged_patch,text_hidden"},
                            *output, error))
        return false;
    return true;
}

} // namespace wevision
