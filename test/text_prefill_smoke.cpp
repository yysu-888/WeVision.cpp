#include "kv_cache.hpp"
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
    if (argc != 5 && argc != 6 && argc != 8 && argc != 9 && argc != 10 &&
        argc != 11) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TOKEN_IDS.i32 VISION_EMBEDDINGS.f32 "
                     "[DEEPSTACK0.f32 DEEPSTACK1.f32 DEEPSTACK2.f32] TRACE_DIR "
                     "[cpu|metal] [GRID_H GRID_W]\n",
                     argv[0]);
        return 2;
    }
    const bool has_deepstack = argc >= 8;
    const bool use_metal =
        (argc == 6 && std::string(argv[5]) == "metal") ||
        ((argc == 9 || argc == 11) && std::string(argv[8]) == "metal");
    const int grid_arg = argc == 11 ? 9 : argc == 10 ? 8 : -1;
    const int grid_h = grid_arg >= 0 ? std::atoi(argv[grid_arg]) : 6;
    const int grid_w = grid_arg >= 0 ? std::atoi(argv[grid_arg + 1]) : 8;
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
    if (!ids_file || !vision_file || image_positions.empty() || grid_h <= 0 ||
        grid_w <= 0 ||
        image_positions.size() != static_cast<size_t>(grid_h * grid_w / 4))
        return 1;
    const int trace_index = has_deepstack ? 7 : 4;
    wevision::GraphSession session(*model, *backend, 512U * 1024U * 1024U);
    // Diagnostic run: capture block boundaries. Production prefill binds only
    // the requested public boundary (normally text.last_hidden).
    wevision::FileTraceSink trace(
        argv[trace_index],
        {"text.block.*", "text.kv.*", "text.last_hidden", "text.next_logits"});
    session.set_trace_sink(&trace);
    ggml_tensor *vision_input =
        ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 2048,
                           static_cast<int64_t>(image_positions.size()));
    ggml_tensor *token_embeddings = nullptr;
    ggml_tensor *embeddings = nullptr;
    ggml_tensor *output = nullptr;
    ggml_tensor *logits = nullptr;
    std::vector<wevision::TextLayerKV> graph_kv;
    if (!session.set_input(vision_input, vision.data(),
                           ggml_nbytes(vision_input), &error) ||
        !wevision::build_text_input_embeddings(session, ids, image_positions,
                                               vision_input, &token_embeddings,
                                               &embeddings, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<ggml_tensor *> deepstack;
    std::vector<std::vector<float>> deepstack_data;
    if (has_deepstack) {
        deepstack_data.resize(3);
        for (int index = 0; index < 3; ++index) {
            deepstack_data[index].resize(2048U * image_positions.size());
            std::ifstream file(argv[4 + index], std::ios::binary);
            file.read(reinterpret_cast<char *>(deepstack_data[index].data()),
                      static_cast<std::streamsize>(
                          deepstack_data[index].size() * sizeof(float)));
            if (!file)
                return 1;
            ggml_tensor *tensor = ggml_new_tensor_2d(
                session.context(), GGML_TYPE_F32, 2048,
                static_cast<int64_t>(image_positions.size()));
            if (!session.set_input(tensor, deepstack_data[index].data(),
                                   ggml_nbytes(tensor), &error))
                return 1;
            deepstack.push_back(tensor);
        }
    }
    wevision::TextPositionIds positions;
    wevision::TextRotaryData rotary;
    if (!wevision::build_text_mrope_positions(ids, 151655, 1, grid_h, grid_w, 2,
                                              &positions, &error) ||
        !wevision::build_text_rotary_data(positions, 128, 5000000.0F,
                                          {24, 20, 20}, &rotary, &error) ||
        !wevision::build_text_prefill(session, embeddings, rotary,
                                      image_positions, deepstack, &graph_kv,
                                      &output, &error) ||
        !wevision::build_text_next_logits(session, output, &logits, &error) ||
        !session.compute(logits, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    wevision::TextKVCache cache;
    if (!cache.initialize(
            model->config.text.layer_count, static_cast<uint32_t>(ids.size()),
            model->config.text.kv_heads, model->config.text.head_dim, &error) ||
        !cache.begin_append(static_cast<uint32_t>(ids.size()), &error) ||
        graph_kv.size() != model->config.text.layer_count)
        return 1;
    const size_t layer_elements = cache.layer_elements();
    std::vector<float> keys(layer_elements), values(layer_elements);
    for (uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        if (!session.copy_to_host(graph_kv[layer].key, keys.data(),
                                  keys.size() * sizeof(float), &error) ||
            !session.copy_to_host(graph_kv[layer].value, values.data(),
                                  values.size() * sizeof(float), &error) ||
            !cache.write_layer(layer, keys.data(), values.data(),
                               layer_elements, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    }
    if (!cache.commit_append(&error) || cache.length() != ids.size())
        return 1;
    return 0;
}
