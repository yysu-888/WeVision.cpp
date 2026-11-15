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
    if (argc != 5)
        return 2;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(v[1], s);
    std::string e;
    auto b =
        m ? wevision::create_backend_context(*m, WEVISION_BACKEND_CPU, s, &e)
          : nullptr;
    if (!b)
        return 1;
    std::vector<float> q, feature;
    if (!read(v[2], &q) || !read(v[3], &feature) || q.size() % 192 ||
        feature.size() != 64U * 512U * 512U)
        return 1;
    wevision::GraphSession x(*m, *b, 256U * 1024U * 1024U);
    wevision::FileTraceSink t(v[4], {"mask.coarse_logits"});
    x.set_trace_sink(&t);
    auto *queries =
        ggml_new_tensor_2d(x.context(), GGML_TYPE_F32, 192, q.size() / 192);
    auto *raw = ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 512, 512, 64);
    if (!x.set_input(queries, q.data(), ggml_nbytes(queries), &e) ||
        !x.set_input(raw, feature.data(), ggml_nbytes(raw), &e))
        return 1;
    auto *pixel =
        ggml_cont(x.context(), ggml_permute(x.context(), raw, 1, 2, 0, 3));
    ggml_tensor *out = nullptr;
    return wevision::build_dynamic_mask_predictor(x, queries, pixel, &out,
                                                  &e) &&
                   x.compute(out, &e)
               ? 0
               : 1;
}
