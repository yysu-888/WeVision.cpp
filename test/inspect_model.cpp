// Model inspection tool. Exercises the internal C++ loader/backend API
// directly: the public C ABI no longer exposes metadata introspection or
// standalone weight upload (wevision_runtime_create() handles upload).
#include "model_loader.hpp"
#include "runtime.hpp"
#include "wevision.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf [--realize=cpu|metal]\n",
                     argv[0]);
        return 2;
    }
    wevision_backend_kind requested_backend = WEVISION_BACKEND_CPU;
    bool realize = false;
    if (argc == 3) {
        realize = true;
        if (std::strcmp(argv[2], "--realize=cpu") == 0) {
            requested_backend = WEVISION_BACKEND_CPU;
        } else if (std::strcmp(argv[2], "--realize=metal") == 0) {
            requested_backend = WEVISION_BACKEND_METAL;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[2]);
            return 2;
        }
    }

    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::unique_ptr<wevision::Model> model =
        wevision::load_model(argv[1], status);
    if (model == nullptr) {
        std::fprintf(stderr, "Cannot load model: %s\n",
                     status == WEVISION_OK ? "unknown error"
                                           : wevision_status_string(status));
        return 1;
    }
    if (status != WEVISION_OK) {
        std::fprintf(stderr, "Model validation failed: %s: %s\n",
                     wevision_status_string(status), model->last_error.c_str());
        return 1;
    }

    const auto &config = model->config;
    std::printf("WeVision GGUF v%u: %zu tensors\n", config.format_version,
                model->tensors.size());
    std::printf("text: %u layers, hidden=%u, heads=%u, kv_heads=%u\n",
                config.text.layer_count, config.text.hidden_size,
                config.text.attention_heads, config.text.kv_heads);
    std::printf("vision: %u layers, hidden=%u\n", config.vision.layer_count,
                config.vision.hidden_size);
    std::printf("mask: %u layers, hidden=%u, <OBJ>=%u\n",
                config.mask.transformer_layers, config.mask.hidden_size,
                config.mask.seg_token_id);

    if (realize) {
        std::string error;
        std::unique_ptr<wevision::BackendContext> backend =
            wevision::create_backend_context(*model, requested_backend, status,
                                             &error);
        if (backend == nullptr) {
            std::fprintf(stderr, "Cannot realize model: %s: %s\n",
                         wevision_status_string(status), error.c_str());
            return 1;
        }
        std::printf("weights: realized on %s\n",
                    requested_backend == WEVISION_BACKEND_METAL ? "Metal"
                                                                : "CPU");
    }
    return 0;
}
