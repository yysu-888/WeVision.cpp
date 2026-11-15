#include "model_loader.hpp"
#include "ms_deform_attn.hpp"
#include "runtime.hpp"
#include "trace.hpp"

#include "ggml.h"

#include <fstream>
#include <string>
#include <vector>

namespace {
bool read(const char *path, std::vector<float> *values) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0);
    if (bytes < 0 || bytes % 4 != 0)
        return false;
    values->resize(static_cast<size_t>(bytes / 4));
    file.read(reinterpret_cast<char *>(values->data()), bytes);
    return static_cast<bool>(file);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7)
        return 2; // MODEL value locations weights TRACE_DIR [cpu|metal]
    const auto kind = argc == 7 && std::string(argv[6]) == "metal"
                          ? WEVISION_BACKEND_METAL
                          : WEVISION_BACKEND_CPU;
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::string error;
    auto model = wevision::load_model(argv[1], status);
    auto backend =
        model ? wevision::create_backend_context(*model, kind, status, &error)
              : nullptr;
    if (!backend)
        return 1;
    std::vector<float> value_data, location_data, weight_data;
    if (!read(argv[2], &value_data) || !read(argv[3], &location_data) ||
        !read(argv[4], &weight_data) || value_data.size() != 2U * 2U * 10U ||
        location_data.size() != 2U * 4U * 2U * 3U ||
        weight_data.size() != 4U * 2U * 3U)
        return 1;
    wevision::GraphSession session(*model, *backend);
    wevision::FileTraceSink trace(argv[5], {"msda.sampler.output"});
    session.set_trace_sink(&trace);
    auto *anchor = ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 4, 3);
    auto *value =
        ggml_new_tensor_3d(session.context(), GGML_TYPE_F32, 2, 2, 10);
    auto *locations =
        ggml_new_tensor_4d(session.context(), GGML_TYPE_F32, 2, 4, 2, 3);
    auto *weights =
        ggml_new_tensor_3d(session.context(), GGML_TYPE_F32, 4, 2, 3);
    std::vector<float> zeros(12, 0.0F);
    if (!session.set_input(anchor, zeros.data(), ggml_nbytes(anchor), &error) ||
        !session.set_input(value, value_data.data(), ggml_nbytes(value),
                           &error) ||
        !session.set_input(locations, location_data.data(),
                           ggml_nbytes(locations), &error) ||
        !session.set_input(weights, weight_data.data(), ggml_nbytes(weights),
                           &error))
        return 1;
    wevision::MSDeformAttnConfig config;
    config.head_dim = 2;
    config.heads = 2;
    config.levels = 2;
    config.points = 2;
    config.heights = {2, 2};
    config.widths = {3, 2};
    ggml_tensor *output = nullptr;
    return wevision::build_ms_deform_attn_sampler(session, anchor, value,
                                                  locations, weights, config,
                                                  &output, &error) &&
                   session.compute(output, &error)
               ? 0
               : 1;
}
