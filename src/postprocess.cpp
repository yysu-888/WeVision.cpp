#include "postprocess.hpp"

#include "parallel.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace wevision {
namespace {

bool parse_number(const std::string &text, size_t begin, size_t end,
                  float *value) {
    if (begin >= end || value == nullptr)
        return false;
    const std::string token = text.substr(begin, end - begin);
    char *tail = nullptr;
    errno = 0;
    const float parsed = std::strtof(token.c_str(), &tail);
    if (errno != 0 || tail == token.c_str() || *tail != '\0' ||
        !std::isfinite(parsed))
        return false;
    *value = parsed;
    return true;
}

void tagged_numbers(const std::string &body, std::vector<float> *values) {
    values->clear();
    for (size_t cursor = 0; cursor < body.size();) {
        const size_t open = body.find('<', cursor);
        if (open == std::string::npos)
            break;
        const size_t close = body.find('>', open + 1);
        if (close == std::string::npos)
            break;
        float value = 0.0F;
        if (parse_number(body, open + 1, close, &value))
            values->push_back(value);
        cursor = close + 1;
    }
}

std::string preceding_label(const std::string &text, size_t before) {
    const size_t close = text.rfind("</ref>", before);
    if (close == std::string::npos)
        return {};
    const size_t open = text.rfind("<ref>", close);
    if (open == std::string::npos || open + 5 > close)
        return {};
    return text.substr(open + 5, close - (open + 5));
}

// Coordinates arrive as <0>..<999> tokens; map them onto [0, extent].
float coordinate(float value, int32_t extent) {
    return std::clamp(value, 0.0F, static_cast<float>(kCoordinateLevels)) *
           static_cast<float>(extent) / static_cast<float>(kCoordinateLevels);
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

} // namespace

bool parse_spatial_response(const std::string &text, int32_t image_width,
                            int32_t image_height, ParsedResponse *output,
                            std::string *error) {
    if (output == nullptr || image_width <= 0 || image_height <= 0) {
        *error = "invalid spatial response destination";
        return false;
    }
    *output = {};
    for (size_t cursor = 0;;) {
        const size_t open = text.find("<box>", cursor);
        if (open == std::string::npos)
            break;
        size_t close = text.find("</box>", open + 5);
        if (close == std::string::npos)
            close = text.size();
        const std::string body = text.substr(open + 5, close - (open + 5));
        std::vector<float> values;
        tagged_numbers(body, &values);
        const std::string label = preceding_label(text, open);
        const bool requests_mask = body.find("<OBJ>") != std::string::npos;
        for (size_t offset = 0; offset + 3 < values.size(); offset += 4) {
            ParsedBox box;
            box.x1 = coordinate(std::min(values[offset], values[offset + 2]),
                                image_width);
            box.y1 = coordinate(
                std::min(values[offset + 1], values[offset + 3]), image_height);
            box.x2 = coordinate(std::max(values[offset], values[offset + 2]),
                                image_width);
            box.y2 = coordinate(
                std::max(values[offset + 1], values[offset + 3]), image_height);
            box.label = label;
            box.requests_mask = requests_mask;
            output->boxes.push_back(std::move(box));
        }
        cursor = close == text.size() ? close : close + 6;
    }
    for (size_t cursor = 0;;) {
        const size_t open = text.find("<point>", cursor);
        if (open == std::string::npos)
            break;
        size_t close = text.find("</point>", open + 7);
        if (close == std::string::npos)
            close = text.size();
        std::vector<float> values;
        tagged_numbers(text.substr(open + 7, close - (open + 7)), &values);
        const std::string label = preceding_label(text, open);
        for (size_t offset = 0; offset + 1 < values.size(); offset += 2) {
            output->points.push_back(
                {coordinate(values[offset], image_width),
                 coordinate(values[offset + 1], image_height), label});
        }
        cursor = close == text.size() ? close : close + 8;
    }
    const size_t count_open = text.find("<count>");
    if (count_open != std::string::npos) {
        const size_t count_close = text.find("</count>", count_open + 7);
        float value = 0.0F;
        if (count_close == std::string::npos ||
            !parse_number(text, count_open + 7, count_close, &value) ||
            value < 0.0F || value > std::numeric_limits<int32_t>::max()) {
            *error = "invalid count in model response";
            return false;
        }
        output->count = static_cast<int32_t>(std::lround(value));
    }
    return true;
}

bool postprocess_mask_logits(const float *logits, int32_t object_count,
                             int32_t source_width, int32_t source_height,
                             int32_t output_width, int32_t output_height,
                             float threshold, std::vector<float> *alpha,
                             std::vector<uint8_t> *mask, std::string *error) {
    if (logits == nullptr || object_count <= 0 || source_width <= 0 ||
        source_height <= 0 || output_width <= 0 || output_height <= 0 ||
        !std::isfinite(threshold) || threshold < 0.0F || threshold > 1.0F ||
        alpha == nullptr || mask == nullptr) {
        *error = "invalid mask postprocess input";
        return false;
    }
    const size_t output_plane =
        static_cast<size_t>(output_width) * output_height;
    alpha->resize(static_cast<size_t>(object_count) * output_plane);
    mask->resize(alpha->size());
    const size_t source_plane =
        static_cast<size_t>(source_width) * source_height;
    std::vector<int32_t> x0_table(static_cast<size_t>(output_width));
    std::vector<int32_t> x1_table(static_cast<size_t>(output_width));
    std::vector<float> wx_table(static_cast<size_t>(output_width));
    for (int32_t x = 0; x < output_width; ++x) {
        const float fx =
            (static_cast<float>(x) + 0.5F) * source_width / output_width -
            0.5F;
        const int32_t x0 = static_cast<int32_t>(std::floor(fx));
        x0_table[static_cast<size_t>(x)] = std::clamp(x0, 0, source_width - 1);
        x1_table[static_cast<size_t>(x)] =
            std::clamp(x0 + 1, 0, source_width - 1);
        wx_table[static_cast<size_t>(x)] = fx - x0;
    }
    const size_t total_rows =
        static_cast<size_t>(object_count) * static_cast<size_t>(output_height);
    parallel_for(total_rows, [&](size_t row_index) {
        const size_t object_index = row_index / static_cast<size_t>(output_height);
        const int32_t y = static_cast<int32_t>(
            row_index % static_cast<size_t>(output_height));
        const float *source = logits + object_index * source_plane;
        const float fy =
            (static_cast<float>(y) + 0.5F) * source_height / output_height -
            0.5F;
        const int32_t y0 = static_cast<int32_t>(std::floor(fy));
        const float wy = fy - y0;
        const float inv_wy = 1.0F - wy;
        const int32_t sy0 = std::clamp(y0, 0, source_height - 1);
        const int32_t sy1 = std::clamp(y0 + 1, 0, source_height - 1);
        const float *row0 = source + static_cast<size_t>(sy0) * source_width;
        const float *row1 = source + static_cast<size_t>(sy1) * source_width;
        const size_t dst_offset =
            object_index * output_plane + static_cast<size_t>(y) * output_width;
        float *alpha_row = alpha->data() + dst_offset;
        uint8_t *mask_row = mask->data() + dst_offset;
        for (int32_t x = 0; x < output_width; ++x) {
            const int32_t sx0 = x0_table[static_cast<size_t>(x)];
            const int32_t sx1 = x1_table[static_cast<size_t>(x)];
            const float wx = wx_table[static_cast<size_t>(x)];
            const float inv_wx = 1.0F - wx;
            const float top = row0[sx0] * inv_wx + row0[sx1] * wx;
            const float bottom = row1[sx0] * inv_wx + row1[sx1] * wx;
            const float value = sigmoid(top * inv_wy + bottom * wy);
            alpha_row[x] = value;
            mask_row[x] = value >= threshold ? 255 : 0;
        }
    });
    return true;
}

} // namespace wevision
