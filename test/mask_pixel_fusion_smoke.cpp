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
    if (argc != 5 && argc != 6)
        return 2;
    const auto kind = argc == 6 && std::string(v[5]) == "metal"
                          ? WEVISION_BACKEND_METAL
                          : WEVISION_BACKEND_CPU;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(v[1], s);
    std::string e;
    auto b = m ? wevision::create_backend_context(*m, kind, s, &e) : nullptr;
    if (!b)
        return 1;
    std::vector<float> fpn, rgb;
    if (!read(v[2], &fpn) || !read(v[3], &rgb) ||
        fpn.size() != 192U * 6U * 8U || rgb.size() != 3U * 1024U * 1024U)
        return 1;
    wevision::GraphSession x(*m, *b, 1536U * 1024U * 1024U);
    wevision::FileTraceSink t(v[4], {"mask.pixel.cnn", "mask.pixel.feature"});
    x.set_trace_sink(&t);
    auto *raw_fpn = ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 8, 6, 192);
    auto *raw_rgb =
        ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 1024, 1024, 3);
    if (!x.set_input(raw_fpn, fpn.data(), ggml_nbytes(raw_fpn), &e) ||
        !x.set_input(raw_rgb, rgb.data(), ggml_nbytes(raw_rgb), &e))
        return 1;
    auto *feature =
        ggml_cont(x.context(), ggml_permute(x.context(), raw_fpn, 1, 2, 0, 3));
    auto *image =
        ggml_cont(x.context(), ggml_permute(x.context(), raw_rgb, 1, 2, 0, 3));
    ggml_tensor *out = nullptr;
    return wevision::build_mask_pixel_fusion(x, feature, image, &out, &e) &&
                   x.compute(out, &e)
               ? 0
               : 1;
}
