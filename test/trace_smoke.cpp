#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"

#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf TRACE_DIR\n", argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::unique_ptr<wevision::Model> model =
        wevision::load_model(argv[1], status);
    if (model == nullptr || status != WEVISION_OK) {
        std::fprintf(stderr, "Cannot load model: %s\n",
                     model != nullptr ? model->last_error.c_str()
                                      : "unknown error");
        return 1;
    }
    std::string error;
    std::unique_ptr<wevision::BackendContext> backend =
        wevision::create_backend_context(*model, WEVISION_BACKEND_CPU, status,
                                         &error);
    if (backend == nullptr || status != WEVISION_OK) {
        std::fprintf(stderr, "Cannot realize CPU backend: %s\n", error.c_str());
        return 1;
    }

    wevision::FileTraceSink trace(argv[2]);
    ggml_tensor *weight =
        backend->tensor("wv.mask.query_builder.content_proj.bias", &error);
    if (weight == nullptr ||
        !trace.watch(
            {"smoke.mask.query_builder.content_proj.bias", "smoke", "hidden"},
            weight, &error) ||
        !trace.capture(*backend, &error)) {
        std::fprintf(stderr, "Cannot write trace: %s\n", error.c_str());
        return 1;
    }
    std::printf("Wrote trace to %s\n", argv[2]);
    return 0;
}
