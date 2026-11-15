// Self-contained drawing toolkit for the CLI example: 5x7 bitmap font,
// rectangles, boxes, points, and the composed prediction overlay.
#include "visualize.hpp"

#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

void blend_channel(unsigned char *destination, unsigned char color,
                   float opacity) {
    *destination = static_cast<unsigned char>(
        std::clamp(*destination * (1.0F - opacity) + color * opacity, 0.0F,
                   255.0F) +
        0.5F);
}

const unsigned char *glyph_rows(unsigned char character) {
    // Compact 5x7 font for visualization labels. Lowercase is rendered as
    // uppercase so the table stays small and dependency-free.
    static constexpr unsigned char blank[7] = {};
    static constexpr unsigned char unknown[7] = {14, 17, 1, 2, 4, 0, 4};
    static constexpr char characters[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:+()[]#%&";
    static constexpr unsigned char glyphs[][7] = {
        {14, 17, 17, 31, 17, 17, 17}, {30, 17, 17, 30, 17, 17, 30},
        {14, 17, 16, 16, 16, 17, 14}, {30, 17, 17, 17, 17, 17, 30},
        {31, 16, 16, 30, 16, 16, 31}, {31, 16, 16, 30, 16, 16, 16},
        {14, 17, 16, 23, 17, 17, 15}, {17, 17, 17, 31, 17, 17, 17},
        {14, 4, 4, 4, 4, 4, 14},      {7, 2, 2, 2, 2, 18, 12},
        {17, 18, 20, 24, 20, 18, 17}, {16, 16, 16, 16, 16, 16, 31},
        {17, 27, 21, 21, 17, 17, 17}, {17, 25, 21, 19, 17, 17, 17},
        {14, 17, 17, 17, 17, 17, 14}, {30, 17, 17, 30, 16, 16, 16},
        {14, 17, 17, 17, 21, 18, 13}, {30, 17, 17, 30, 20, 18, 17},
        {15, 16, 16, 14, 1, 1, 30},   {31, 4, 4, 4, 4, 4, 4},
        {17, 17, 17, 17, 17, 17, 14}, {17, 17, 17, 17, 17, 10, 4},
        {17, 17, 17, 21, 21, 21, 10}, {17, 17, 10, 4, 10, 17, 17},
        {17, 17, 10, 4, 4, 4, 4},     {31, 1, 2, 4, 8, 16, 31},
        {14, 17, 19, 21, 25, 17, 14}, {4, 12, 4, 4, 4, 4, 14},
        {14, 17, 1, 2, 4, 8, 31},     {30, 1, 1, 14, 1, 1, 30},
        {2, 6, 10, 18, 31, 2, 2},     {31, 16, 16, 30, 1, 1, 30},
        {14, 16, 16, 30, 17, 17, 14}, {31, 1, 2, 4, 8, 8, 8},
        {14, 17, 17, 14, 17, 17, 14}, {14, 17, 17, 15, 1, 1, 14},
        {0, 0, 0, 31, 0, 0, 0},       {0, 0, 0, 0, 0, 0, 4},
        {0, 0, 0, 0, 0, 0, 31},       {1, 2, 2, 4, 8, 8, 16},
        {0, 4, 4, 31, 4, 4, 0},       {2, 4, 8, 8, 8, 4, 2},
        {8, 4, 2, 2, 2, 4, 8},        {14, 8, 8, 8, 8, 8, 14},
        {14, 2, 2, 2, 2, 2, 14},      {10, 31, 10, 10, 31, 10, 0},
        {17, 2, 4, 8, 17, 0, 0},      {12, 18, 20, 8, 21, 18, 13},
    };
    if (character == ' ')
        return blank;
    character = static_cast<unsigned char>(std::toupper(character));
    const char *position = std::strchr(characters, character);
    return position == nullptr ? unknown : glyphs[position - characters];
}

void paint_pixel(std::vector<unsigned char> *image, int width, int height,
                 int x, int y, const unsigned char color[3]) {
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;
    unsigned char *pixel =
        image->data() + (static_cast<size_t>(y) * width + x) * 3U;
    pixel[0] = color[0];
    pixel[1] = color[1];
    pixel[2] = color[2];
}

void fill_rect(std::vector<unsigned char> *image, int width, int height, int x1,
               int y1, int x2, int y2, const unsigned char color[3]) {
    x1 = std::clamp(x1, 0, width);
    y1 = std::clamp(y1, 0, height);
    x2 = std::clamp(x2, 0, width);
    y2 = std::clamp(y2, 0, height);
    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            paint_pixel(image, width, height, x, y, color);
        }
    }
}

std::string display_label(const char *label, const std::string &fallback) {
    const std::string source =
        label != nullptr && label[0] != '\0' ? std::string(label) : fallback;
    std::string result;
    for (size_t index = 0; index < source.size();) {
        const unsigned char value = static_cast<unsigned char>(source[index]);
        if (value < 0x80U) {
            if (value >= 0x20U && value <= 0x7eU)
                result.push_back(value);
            ++index;
            continue;
        }
        result.push_back('?');
        ++index;
        while (index < source.size() &&
               (static_cast<unsigned char>(source[index]) & 0xc0U) == 0x80U) {
            ++index;
        }
    }
    return result;
}

void draw_label(std::vector<unsigned char> *image, int width, int height,
                int anchor_x, int anchor_y, const std::string &text,
                const unsigned char color[3]) {
    if (text.empty())
        return;
    const int scale = std::max(1, std::min(width, height) / 480);
    const int padding = std::max(2, scale);
    const int advance = 6 * scale;
    const int max_characters = std::max(1, (width - 2 * padding) / advance);
    std::string visible = text.substr(0, static_cast<size_t>(max_characters));
    const int text_width = static_cast<int>(visible.size()) * advance - scale;
    const int text_height = 7 * scale;
    const int box_width = text_width + padding * 2;
    const int box_height = text_height + padding * 2;
    int left = std::clamp(anchor_x, 0, std::max(0, width - box_width));
    int top = anchor_y - box_height;
    if (top < 0)
        top = std::clamp(anchor_y, 0, std::max(0, height - box_height));
    static constexpr unsigned char background[3] = {16, 16, 16};
    static constexpr unsigned char foreground[3] = {255, 255, 255};
    fill_rect(image, width, height, left, top, left + box_width,
              top + box_height, background);
    fill_rect(image, width, height, left, top, left + std::max(2, scale),
              top + box_height, color);
    int cursor = left + padding;
    for (unsigned char character : visible) {
        const unsigned char *rows = glyph_rows(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[row] & (1U << (4 - column))) == 0)
                    continue;
                fill_rect(image, width, height, cursor + column * scale,
                          top + padding + row * scale,
                          cursor + (column + 1) * scale,
                          top + padding + (row + 1) * scale, foreground);
            }
        }
        cursor += advance;
    }
}

void draw_box(std::vector<unsigned char> *image, int width, int height,
              const wevision_box_result &box, const unsigned char color[3]) {
    const int x1 = std::clamp(static_cast<int>(std::lround(box.x1)), 0,
                              std::max(width - 1, 0));
    const int y1 = std::clamp(static_cast<int>(std::lround(box.y1)), 0,
                              std::max(height - 1, 0));
    const int x2 = std::clamp(static_cast<int>(std::lround(box.x2)), 0,
                              std::max(width - 1, 0));
    const int y2 = std::clamp(static_cast<int>(std::lround(box.y2)), 0,
                              std::max(height - 1, 0));
    const int thickness = std::max(2, std::min(width, height) / 300);
    auto paint = [&](int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return;
        unsigned char *pixel =
            image->data() + (static_cast<size_t>(y) * width + x) * 3U;
        pixel[0] = color[0];
        pixel[1] = color[1];
        pixel[2] = color[2];
    };
    for (int offset = 0; offset < thickness; ++offset) {
        for (int x = x1; x <= x2; ++x) {
            paint(x, y1 + offset);
            paint(x, y2 - offset);
        }
        for (int y = y1; y <= y2; ++y) {
            paint(x1 + offset, y);
            paint(x2 - offset, y);
        }
    }
}

void draw_point(std::vector<unsigned char> *image, int width, int height,
                const wevision_point_result &point,
                const unsigned char color[3]) {
    const int center_x = std::clamp(static_cast<int>(std::lround(point.x)), 0,
                                    std::max(width - 1, 0));
    const int center_y = std::clamp(static_cast<int>(std::lround(point.y)), 0,
                                    std::max(height - 1, 0));
    const int radius = std::max(4, std::min(width, height) / 120);
    static constexpr unsigned char outline[3] = {255, 255, 255};
    for (int y = -radius - 2; y <= radius + 2; ++y) {
        for (int x = -radius - 2; x <= radius + 2; ++x) {
            const int distance = x * x + y * y;
            if (distance <= (radius + 2) * (radius + 2)) {
                paint_pixel(image, width, height, center_x + x, center_y + y,
                            distance <= radius * radius ? color : outline);
            }
        }
    }
}

} // namespace

bool write_visualization(const std::string &path, const wevision_result &result,
                         const std::string &fallback_label,
                         const unsigned char *rgb, int width, int height) {
    if (path.empty())
        return true;
    if (!rgb || width <= 0 || height <= 0)
        return false;
    static constexpr unsigned char colors[][3] = {
        {255, 72, 72},  {72, 220, 120}, {72, 144, 255},  {255, 196, 72},
        {204, 96, 255}, {72, 224, 224}, {255, 128, 192}, {176, 224, 72},
    };
    std::vector<unsigned char> visual(rgb, rgb + static_cast<size_t>(width) *
                                                     height * 3U);
    const bool mask_shape_matches =
        result.image_width == width && result.image_height == height;
    const size_t pixels = static_cast<size_t>(width) * height;
    for (uint32_t index = 0; index < result.box_count; ++index) {
        const auto &box = result.boxes[index];
        const unsigned char *color =
            colors[index % (sizeof(colors) / sizeof(colors[0]))];
        if (mask_shape_matches &&
            (box.alpha != nullptr || box.mask != nullptr)) {
            for (size_t pixel_index = 0; pixel_index < pixels; ++pixel_index) {
                const float coverage =
                    box.alpha != nullptr
                        ? std::clamp(box.alpha[pixel_index], 0.0F, 1.0F)
                        : box.mask[pixel_index] / 255.0F;
                const float opacity = coverage * 0.45F;
                unsigned char *pixel = visual.data() + pixel_index * 3U;
                blend_channel(pixel, color[0], opacity);
                blend_channel(pixel + 1, color[1], opacity);
                blend_channel(pixel + 2, color[2], opacity);
            }
        }
    }
    for (uint32_t index = 0; index < result.box_count; ++index) {
        draw_box(&visual, width, height, result.boxes[index],
                 colors[index % (sizeof(colors) / sizeof(colors[0]))]);
    }
    for (uint32_t index = 0; index < result.point_count; ++index) {
        draw_point(&visual, width, height, result.points[index],
                   colors[index % (sizeof(colors) / sizeof(colors[0]))]);
    }
    for (uint32_t index = 0; index < result.box_count; ++index) {
        const auto &box = result.boxes[index];
        const unsigned char *color =
            colors[index % (sizeof(colors) / sizeof(colors[0]))];
        draw_label(&visual, width, height,
                   static_cast<int>(std::lround(box.x1)),
                   static_cast<int>(std::lround(box.y1)),
                   display_label(box.label, fallback_label), color);
    }
    for (uint32_t index = 0; index < result.point_count; ++index) {
        const auto &point = result.points[index];
        const unsigned char *color =
            colors[index % (sizeof(colors) / sizeof(colors[0]))];
        const int offset = std::max(8, std::min(width, height) / 100);
        draw_label(&visual, width, height,
                   static_cast<int>(std::lround(point.x)) + offset,
                   static_cast<int>(std::lround(point.y)),
                   display_label(point.label, fallback_label), color);
    }
    if (!stbi_write_png(path.c_str(), width, height, 3, visual.data(),
                        width * 3))
        return false;
    std::printf("visualization=%s\n", path.c_str());
    return true;
}
