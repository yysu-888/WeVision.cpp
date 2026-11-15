#include "ggml.h"
#include "model_loader.hpp"
#include "pixel_ops.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include <fstream>
#include <memory>
#include <vector>

int main(int argc, char **v) {
    if (argc < 4 || argc > 5)
        return 2;
    const auto kind = argc == 5 && std::string(v[4]) == "metal"
                          ? WEVISION_BACKEND_METAL
                          : WEVISION_BACKEND_CPU;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(v[1], s);
    std::string e;
    auto b = m ? wevision::create_backend_context(*m, kind, s, &e) : nullptr;
    if (!b)
        return 1;
    std::vector<float> a(8 * 3 * 2);
    std::ifstream f(v[2], std::ios::binary);
    f.read((char *)a.data(), a.size() * 4);
    if (!f)
        return 1;
    wevision::GraphSession x(*m, *b);
    wevision::FileTraceSink t(v[3],
                              {"pixel.shuffle", "pixel.resize", "pixel.pool"});
    x.set_trace_sink(&t);
    auto *in = ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 8, 3, 2);
    if (!x.set_input(in, a.data(), ggml_nbytes(in), &e))
        return 1;
    ggml_tensor *sh = nullptr, *rs = nullptr, *pool = nullptr;
    if (!wevision::build_pixel_shuffle2x(x, in, &sh, &e) ||
        !x.checkpoint(
            {"pixel.shuffle", "pixel", "B,C,H,W"},
            ggml_cont(x.context(), ggml_permute(x.context(), sh, 2, 0, 1, 3)),
            &e) ||
        !wevision::build_bilinear_resize(x, sh, 5, 3, &rs, &e) ||
        !x.checkpoint(
            {"pixel.resize", "pixel", "B,C,H,W"},
            ggml_cont(x.context(), ggml_permute(x.context(), rs, 2, 0, 1, 3)),
            &e) ||
        !wevision::build_adaptive_avg_pool2d(x, sh, 5, 3, &pool, &e) ||
        !x.checkpoint(
            {"pixel.pool", "pixel", "B,C,H,W"},
            ggml_cont(x.context(), ggml_permute(x.context(), pool, 2, 0, 1, 3)),
            &e) ||
        !x.compute(rs, &e))
        return 1;
    return 0;
}
