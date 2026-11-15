#include "ggml.h"
#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include <fstream>
#include <memory>
#include <vector>

namespace {
bool read(const char *p, std::vector<float> *v) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        return false;
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    f.seekg(0);
    if (n < 0 || n % 4)
        return false;
    v->resize((size_t)n / 4);
    f.read((char *)v->data(), n);
    return (bool)f;
}
} // namespace

int main(int argc, char **v) {
    if (argc != 9 && argc != 10)
        return 2;
    const auto kind = argc == 10 && std::string(v[9]) == "metal"
                          ? WEVISION_BACKEND_METAL
                          : WEVISION_BACKEND_CPU;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(v[1], s);
    std::string e;
    auto b = m ? wevision::create_backend_context(*m, kind, s, &e) : nullptr;
    if (!b)
        return 1;
    std::vector<std::vector<float>> features(4,
                                             std::vector<float>(1024 * 6 * 8));
    for (int i = 0; i < 4; ++i)
        if (!read(v[2 + i], &features[i]))
            return 1;
    std::vector<float> hidden, boxes;
    if (!read(v[6], &hidden) || !read(v[7], &boxes) || hidden.size() % 2048 ||
        boxes.size() != hidden.size() / 2048 * 4)
        return 1;
    wevision::GraphSession x(*m, *b, 512U * 1024U * 1024U);
    wevision::FileTraceSink t(v[8], {"mask.query.*", "mask.transformer.*"});
    x.set_trace_sink(&t);
    std::vector<ggml_tensor *> in;
    for (int i = 0; i < 4; ++i) {
        auto *q = ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 8, 6, 1024);
        if (!x.set_input(q, features[i].data(), ggml_nbytes(q), &e))
            return 1;
        in.push_back(
            ggml_cont(x.context(), ggml_permute(x.context(), q, 1, 2, 0, 3)));
    }
    auto *q = ggml_new_tensor_2d(x.context(), GGML_TYPE_F32, 2048,
                                 hidden.size() / 2048);
    if (!x.set_input(q, hidden.data(), ggml_nbytes(q), &e))
        return 1;
    wevision::MaskProjectedFeatures p;
    wevision::MaskEncoderInputs i;
    wevision::MaskQueryOutput query;
    ggml_tensor *memory = nullptr, *out = nullptr;
    if (!wevision::build_mask_encoder_projection(x, in, &p, &e) ||
        !wevision::build_mask_encoder_inputs(x, p, &i, &e) ||
        !wevision::build_mask_spatial_encoder(x, i, 3, &memory, &e) ||
        !wevision::build_mask_query(x, q, boxes, &query, &e) ||
        !wevision::build_mask_transformer(x, query.content, query.position,
                                          boxes, memory, i, &out, &e) ||
        !x.compute(out, &e))
        return 1;
    return 0;
}
