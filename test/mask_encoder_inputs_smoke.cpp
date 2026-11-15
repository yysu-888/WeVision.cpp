#include "ggml.h"
#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include <fstream>
#include <memory>
#include <vector>

int main(int argc, char **v) {
    if (argc != 7)
        return 2;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(v[1], s);
    std::string e;
    auto b =
        m ? wevision::create_backend_context(*m, WEVISION_BACKEND_CPU, s, &e)
          : nullptr;
    if (!b)
        return 1;
    wevision::GraphSession x(*m, *b);
    wevision::FileTraceSink t(v[6], {"mask.visual_encoder.src",
                                     "mask.visual_encoder.pos",
                                     "mask.visual_encoder.memory"});
    x.set_trace_sink(&t);
    std::vector<std::vector<float>> data(4, std::vector<float>(1024 * 6 * 8));
    std::vector<ggml_tensor *> in;
    for (int i = 0; i < 4; ++i) {
        std::ifstream f(v[2 + i], std::ios::binary);
        f.read((char *)data[i].data(), data[i].size() * 4);
        if (!f)
            return 1;
        auto *q = ggml_new_tensor_3d(x.context(), GGML_TYPE_F32, 8, 6, 1024);
        if (!x.set_input(q, data[i].data(), ggml_nbytes(q), &e))
            return 1;
        in.push_back(
            ggml_cont(x.context(), ggml_permute(x.context(), q, 1, 2, 0, 3)));
    }
    wevision::MaskProjectedFeatures p;
    wevision::MaskEncoderInputs o;
    ggml_tensor *memory = nullptr;
    if (!wevision::build_mask_encoder_projection(x, in, &p, &e) ||
        !wevision::build_mask_encoder_inputs(x, p, &o, &e) ||
        !wevision::build_mask_spatial_encoder(x, o, 3, &memory, &e) ||
        !x.compute(memory, &e))
        return 1;
    return 0;
}
