#include "model_loader.hpp"
#include "runtime.hpp"
#include "text.hpp"
#include "trace.hpp"

#include "ggml.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TOKEN_IDS.i32 VISION_EMBEDDINGS.f32 "
                     "TRACE_DIR\n",
                     argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    std::string error;
    auto backend = model == nullptr
                       ? nullptr
                       : wevision::create_backend_context(
                             *model, WEVISION_BACKEND_CPU, status, &error);
    if (backend == nullptr) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::ifstream ids_file(argv[2], std::ios::binary);
    ids_file.seekg(0, std::ios::end);
    const auto bytes = ids_file.tellg();
    ids_file.seekg(0);
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0)
        return 1;
    std::vector<int32_t> ids(static_cast<size_t>(
        bytes / static_cast<std::streamoff>(sizeof(int32_t))));
    ids_file.read(reinterpret_cast<char *>(ids.data()), bytes);
    std::vector<int32_t> image_positions;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == 151655)
            image_positions.push_back(static_cast<int32_t>(i));
    std::vector<float> vision(2048U * image_positions.size());
    std::ifstream vision_file(argv[3], std::ios::binary);
    vision_file.read(
        reinterpret_cast<char *>(vision.data()),
        static_cast<std::streamsize>(vision.size() * sizeof(float)));
    if (!ids_file || !vision_file || image_positions.size() != 12)
        return 1;
    wevision::GraphSession session(*model, *backend, 64U * 1024U * 1024U);
    wevision::FileTraceSink trace(argv[4], {"text.block.00.output"});
    session.set_trace_sink(&trace);
    ggml_tensor *vision_input =
        ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 2048,
                           static_cast<int64_t>(image_positions.size()));
    ggml_tensor *token_embeddings = nullptr;
    ggml_tensor *embeddings = nullptr;
    ggml_tensor *output = nullptr;
    if (!session.set_input(vision_input, vision.data(),
                           ggml_nbytes(vision_input), &error) ||
        !wevision::build_text_input_embeddings(session, ids, image_positions,
                                               vision_input, &token_embeddings,
                                               &embeddings, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    wevision::TextPositionIds positions;
    wevision::TextRotaryData rotary;
    if (!wevision::build_text_mrope_positions(ids, 151655, 1, 6, 8, 2,
                                              &positions, &error) ||
        !wevision::build_text_rotary_data(positions, 128, 5000000.0F,
                                          {24, 20, 20}, &rotary, &error) ||
        !wevision::build_text_decoder_block(session, 0, embeddings, rotary,
                                            nullptr, nullptr, &output,
                                            &error) ||
        !session.compute(output, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
