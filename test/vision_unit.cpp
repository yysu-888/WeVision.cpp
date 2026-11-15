#include "vision.hpp"

#include <cmath>
#include <cstdio>

namespace {

bool close(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-7F;
}

} // namespace

int main() {
    wevision::VisionPreprocessConfig config;
    config.min_pixels = 16384;
    config.max_pixels = 16384;
    wevision::SmartResizeShape shape;
    if (!wevision::smart_resize(1080, 1620, config, &shape) ||
        shape.height != 96 || shape.width != 128) {
        std::fprintf(stderr, "smart_resize mismatch: %dx%d\n", shape.height,
                     shape.width);
        return 1;
    }

    wevision::RgbImage image;
    image.width = 64;
    image.height = 32;
    image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3U);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const size_t offset =
                (static_cast<size_t>(y) * image.width + x) * 3U;
            image.pixels[offset + 0] =
                static_cast<uint8_t>((y * image.width + x) % 256);
            image.pixels[offset + 1] = 127;
            image.pixels[offset + 2] = 255;
        }
    }
    config.min_pixels = 1;
    config.max_pixels = 1 << 28;
    wevision::PatchifiedImage patches;
    if (!wevision::patchify_resized_rgb(image, config, &patches) ||
        patches.grid_t != 1 || patches.grid_h != 2 || patches.grid_w != 4 ||
        patches.patch_dim != 1536 || patches.values.size() != 8U * 1536U) {
        std::fprintf(stderr, "patchify shape mismatch\n");
        return 1;
    }
    // Sequence index 4 is [block_y=0, block_x=1, intra_y=0, intra_x=0],
    // proving merge-local patch ordering rather than naive row-major reshape.
    const size_t sequence_four = static_cast<size_t>(4) * patches.patch_dim;
    const float expected_red = (32.0F / 255.0F - 0.5F) / 0.5F;
    const float expected_green = (127.0F / 255.0F - 0.5F) / 0.5F;
    if (!close(patches.values[0], -1.0F) ||
        !close(patches.values[256], -1.0F) ||
        !close(patches.values[512], expected_green) ||
        !close(patches.values[1024], 1.0F) ||
        !close(patches.values[sequence_four], expected_red)) {
        std::fprintf(stderr, "patchify ordering or normalization mismatch\n");
        return 1;
    }

    wevision::RgbImage tiny;
    tiny.width = 2;
    tiny.height = 2;
    tiny.pixels = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110};
    wevision::MaskImageInput mask;
    if (!wevision::prepare_mask_image(tiny, 2, &mask) ||
        mask.original_width != 2 || mask.original_height != 2 ||
        mask.rgb.size() != 12U || !close(mask.rgb[0], 0.0F) ||
        !close(mask.rgb[4], 10.0F / 255.0F) ||
        !close(mask.rgb[8], 20.0F / 255.0F)) {
        std::fprintf(stderr, "mask canvas layout mismatch\n");
        return 1;
    }
    config.min_pixels = 16384;
    config.max_pixels = 16384;
    wevision::VisionImageInput prepared;
    if (!wevision::prepare_vision_image(image, config, &prepared) ||
        prepared.grid_thw[0] != 1 || prepared.grid_thw[1] != 6 ||
        prepared.grid_thw[2] != 12 ||
        prepared.patches.values.size() != 72U * 1536U) {
        std::fprintf(stderr, "vision input preparation mismatch\n");
        return 1;
    }
    wevision::VisionPositionData positions;
    if (!wevision::build_vision_position_data(6, 8, 2, 48, &positions) ||
        positions.token_count != 48 || positions.axial_hw.size() != 96U ||
        positions.interpolation_ids.size() != 192U ||
        positions.interpolation_weights.size() != 192U ||
        positions.axial_hw[0] != 0 || positions.axial_hw[1] != 0 ||
        positions.axial_hw[8] != 0 || positions.axial_hw[9] != 2 ||
        positions.axial_hw[32] != 2 || positions.axial_hw[33] != 0 ||
        positions.interpolation_ids[0] != 0 ||
        positions.interpolation_ids[3] != 49 ||
        !close(positions.interpolation_weights[0], 1.0F) ||
        !close(positions.interpolation_weights[1] +
                   positions.interpolation_weights[0],
               1.0F)) {
        std::fprintf(stderr, "vision position layout mismatch\n");
        return 1;
    }
    wevision::VisionRotaryData rotary;
    if (!wevision::build_vision_rotary_data(positions, 64, 10000.0F, &rotary) ||
        rotary.cos.size() != 48U * 64U || rotary.sin.size() != 48U * 64U ||
        !close(rotary.cos[0], 1.0F) || !close(rotary.sin[0], 0.0F) ||
        !close(rotary.cos[64 + 16], std::cos(1.0F)) ||
        !close(rotary.cos[64 + 48], std::cos(1.0F))) {
        std::fprintf(stderr, "vision rotary layout mismatch\n");
        return 1;
    }
    std::puts("vision preprocessing unit checks passed");
    return 0;
}
