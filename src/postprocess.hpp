#ifndef WEVISION_POSTPROCESS_HPP
#define WEVISION_POSTPROCESS_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace wevision {

struct ParsedBox {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    std::string label;
    bool requests_mask = false;
};

struct ParsedPoint {
    float x = 0.0F;
    float y = 0.0F;
    std::string label;
};

struct ParsedResponse {
    std::vector<ParsedBox> boxes;
    std::vector<ParsedPoint> points;
    int32_t count = -1;
};

bool parse_spatial_response(const std::string &text, int32_t image_width,
                            int32_t image_height, ParsedResponse *output,
                            std::string *error);

// Converts object-major logits at the model canvas size to row-major masks at
// the original image size. Alpha is sigmoid(logit); mask uses alpha >=
// threshold.
bool postprocess_mask_logits(const float *logits, int32_t object_count,
                             int32_t source_width, int32_t source_height,
                             int32_t output_width, int32_t output_height,
                             float threshold, std::vector<float> *alpha,
                             std::vector<uint8_t> *mask, std::string *error);

} // namespace wevision

#endif
