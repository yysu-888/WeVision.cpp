#include "postprocess.hpp"

#include <cmath>
#include <string>
#include <vector>

int main() {
    std::string error;
    wevision::ParsedResponse parsed;
    if (!wevision::parse_spatial_response(
            "<ref>person</ref><box><750><1000><10><71><OBJ></box>"
            "<point><500><250></point><count>3</count>",
            200, 100, &parsed, &error))
        return 1;
    if (parsed.boxes.size() != 1 || parsed.points.size() != 1 ||
        parsed.count != 3 || parsed.boxes[0].label != "person" ||
        !parsed.boxes[0].requests_mask ||
        std::fabs(parsed.boxes[0].x1 - 2.0F) > 1e-6F ||
        std::fabs(parsed.boxes[0].y2 - 100.0F) > 1e-6F ||
        std::fabs(parsed.points[0].x - 100.0F) > 1e-6F)
        return 1;
    const float logits[] = {-2.0F, 2.0F, -2.0F, 2.0F};
    std::vector<float> alpha;
    std::vector<uint8_t> mask;
    if (!wevision::postprocess_mask_logits(logits, 1, 2, 2, 4, 4, 0.5F, &alpha,
                                           &mask, &error) ||
        alpha.size() != 16 || mask.size() != 16 || mask.front() != 0 ||
        mask.back() != 255)
        return 1;

    const float multi_logits[] = {-2.0F, 2.0F,  -2.0F, 2.0F,
                                  2.0F,  -2.0F, 2.0F,  -2.0F};
    if (!wevision::postprocess_mask_logits(multi_logits, 2, 2, 2, 4, 4, 0.5F,
                                           &alpha, &mask, &error) ||
        alpha.size() != 32 || mask.size() != 32 || mask.front() != 0 ||
        mask[15] != 255 || mask[16] != 255 || mask.back() != 0)
        return 1;
    return 0;
}
