#include "model_loader.hpp"
#include "runtime.hpp"
#include "text.hpp"
#include "trace.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TOKEN_IDS.i32 VISION_EMBEDDINGS.f32 "
                     "TRACE_DIR [cpu|metal]\n",
                     argv[0]);
        return 2;
    }
    const bool use_metal = argc == 6 && std::string(argv[5]) == "metal";
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    std::string error;
    auto backend = model == nullptr ? nullptr
                                    : wevision::create_backend_context(
                                          *model,
                                          use_metal ? WEVISION_BACKEND_METAL
                                                    : WEVISION_BACKEND_CPU,
                                          status, &error);
    if (backend == nullptr) {
        std::fprintf(stderr, "Backend: %s\n", error.c_str());
        return 1;
    }
    std::ifstream ids_file(argv[2], std::ios::binary);
    ids_file.seekg(0, std::ios::end);
    const auto bytes = ids_file.tellg();
    ids_file.seekg(0);
    if (bytes <= 0 ||
        bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
        std::fputs("Invalid token id payload\n", stderr);
        return 1;
    }
    std::vector<int32_t> ids(static_cast<size_t>(
        bytes / static_cast<std::streamoff>(sizeof(int32_t))));
    ids_file.read(reinterpret_cast<char *>(ids.data()), bytes);
    if (!ids_file) {
        std::fputs("Invalid token id payload\n", stderr);
        return 1;
    }
    std::vector<int32_t> image_positions;
    for (size_t index = 0; index < ids.size(); ++index)
        if (ids[index] == 151655)
            image_positions.push_back(static_cast<int32_t>(index));
    if (image_positions.size() != 12) {
        std::fprintf(stderr, "Expected 12 image-pad positions, found %zu\n",
                     image_positions.size());
        return 1;
    }
    std::vector<float> vision(2048U * image_positions.size());
    std::ifstream vision_file(argv[3], std::ios::binary);
    vision_file.read(
        reinterpret_cast<char *>(vision.data()),
        static_cast<std::streamsize>(vision.size() * sizeof(float)));
    if (!vision_file) {
        std::fputs("Invalid vision embedding payload\n", stderr);
        return 1;
    }
    wevision::GraphSession session(*model, *backend, 24U * 1024U * 1024U);
    wevision::FileTraceSink trace(argv[4], {"text.*"});
    session.set_trace_sink(&trace);
    ggml_tensor *vision_input =
        ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 2048,
                           static_cast<int64_t>(image_positions.size()));
    if (!session.set_input(vision_input, vision.data(),
                           ggml_nbytes(vision_input), &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *base = nullptr;
    ggml_tensor *injected = nullptr;
    if (!wevision::build_text_input_embeddings(session, ids, image_positions,
                                               vision_input, &base, &injected,
                                               &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!session.compute(injected, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
