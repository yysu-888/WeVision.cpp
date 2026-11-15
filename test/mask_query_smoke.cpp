#include "ggml.h"
#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5)
        return 2;
    wevision_status s = WEVISION_ERROR_INTERNAL;
    auto m = wevision::load_model(argv[1], s);
    std::string e;
    auto b =
        m ? wevision::create_backend_context(*m, WEVISION_BACKEND_CPU, s, &e)
          : nullptr;
    if (!b)
        return 1;
    std::vector<float> h(2048), box(4);
    std::ifstream a(argv[2], std::ios::binary), c(argv[3], std::ios::binary);
    a.read((char *)h.data(), h.size() * 4);
    c.read((char *)box.data(), box.size() * 4);
    if (!a || !c)
        return 1;
    wevision::GraphSession x(*m, *b);
    wevision::FileTraceSink t(argv[4], {"mask.query.*"});
    x.set_trace_sink(&t);
    auto *in = ggml_new_tensor_2d(x.context(), GGML_TYPE_F32, 2048, 1);
    wevision::MaskQueryOutput o;
    if (!x.set_input(in, h.data(), ggml_nbytes(in), &e) ||
        !wevision::build_mask_query(x, in, box, &o, &e) ||
        !x.compute(o.content, &e))
        return 1;
    return 0;
}
