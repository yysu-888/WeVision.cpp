#ifndef WEVISION_INFERENCE_HPP
#define WEVISION_INFERENCE_HPP

#include "postprocess.hpp"
#include "vision.hpp"
#include "wevision.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wevision {

class BackendContext;
class QwenTokenizer;
struct Model;

struct InferenceOutput {
    std::string text;
    ParsedResponse parsed;
    int32_t image_width = 0;
    int32_t image_height = 0;
    // Object-major, original-image-sized planes. Empty for non-mask tasks.
    std::vector<float> alpha;
    std::vector<uint8_t> mask;
    wevision_metrics metrics{};
};

bool run_inference(const Model &model, const BackendContext &backend,
                   const QwenTokenizer &tokenizer, size_t graph_memory_bytes,
                   uint32_t runtime_max_new_tokens,
                   const wevision_request &request, InferenceOutput *output,
                   std::string *error);

} // namespace wevision

#endif
