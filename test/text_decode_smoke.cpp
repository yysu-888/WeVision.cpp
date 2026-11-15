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

namespace {

bool read_f32(const char *path, std::vector<float> *output) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        return false;
    output->resize(static_cast<size_t>(
        bytes / static_cast<std::streamoff>(sizeof(float))));
    file.read(reinterpret_cast<char *>(output->data()), bytes);
    return static_cast<bool>(file);
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 9) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TOKEN_IDS.i32 VISION.f32 DEEP0.f32 "
                     "DEEP1.f32 DEEP2.f32 OUTPUT_DIR NEXT_TOKEN_ID\n",
                     argv[0]);
        return 2;
    }
    const int32_t next_token = std::stoi(argv[8]);
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
    std::vector<float> vision;
    if (!read_f32(argv[3], &vision) ||
        vision.size() != 2048U * image_positions.size())
        return 1;
    std::vector<std::vector<float>> deep(3);
    for (int i = 0; i < 3; ++i)
        if (!read_f32(argv[4 + i], &deep[i]) || deep[i].size() != vision.size())
            return 1;
    wevision::TextKVCache cache;
    wevision::TextPositionIds positions;
    wevision::TextRotaryData rotary;
    if (!wevision::build_text_mrope_positions(ids, 151655, 1, 6, 8, 2,
                                              &positions, &error) ||
        !wevision::build_text_rotary_data(positions, 128, 5000000.0F,
                                          {24, 20, 20}, &rotary, &error) ||
        !cache.initialize(model->config.text.layer_count,
                          static_cast<uint32_t>(ids.size() + 8),
                          model->config.text.kv_heads,
                          model->config.text.head_dim, &error))
        return 1;
    {
        wevision::GraphSession session(*model, *backend, 512U * 1024U * 1024U);
        wevision::FileTraceSink trace(std::string(argv[7]) + ".prefill",
                                      {"text.kv.*"});
        session.set_trace_sink(&trace);
        ggml_tensor *vision_input =
            ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 2048,
                               static_cast<int64_t>(image_positions.size()));
        ggml_tensor *token = nullptr;
        ggml_tensor *embedded = nullptr;
        ggml_tensor *hidden = nullptr;
        ggml_tensor *logits = nullptr;
        if (!session.set_input(vision_input, vision.data(),
                               ggml_nbytes(vision_input), &error) ||
            !wevision::build_text_input_embeddings(
                session, ids, image_positions, vision_input, &token, &embedded,
                &error))
            return 1;
        std::vector<ggml_tensor *> deep_inputs;
        for (int i = 0; i < 3; ++i) {
            ggml_tensor *tensor = ggml_new_tensor_2d(
                session.context(), GGML_TYPE_F32, 2048,
                static_cast<int64_t>(image_positions.size()));
            if (!session.set_input(tensor, deep[i].data(), ggml_nbytes(tensor),
                                   &error))
                return 1;
            deep_inputs.push_back(tensor);
        }
        std::vector<wevision::TextLayerKV> graph_kv;
        if (!wevision::build_text_prefill(session, embedded, rotary,
                                          image_positions, deep_inputs,
                                          &graph_kv, &hidden, &error) ||
            !wevision::build_text_next_logits(session, hidden, &logits,
                                              &error) ||
            !session.compute(logits, &error) ||
            !cache.begin_append(static_cast<uint32_t>(ids.size()), &error))
            return 1;
        const size_t prefill_elements = static_cast<size_t>(ids.size()) *
                                        cache.kv_heads() * cache.head_dim();
        std::vector<float> keys(prefill_elements), values(prefill_elements);
        for (uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            if (!session.copy_to_host(graph_kv[layer].key, keys.data(),
                                      keys.size() * sizeof(float), &error) ||
                !session.copy_to_host(graph_kv[layer].value, values.data(),
                                      values.size() * sizeof(float), &error) ||
                !cache.write_layer(layer, keys.data(), values.data(),
                                   prefill_elements, &error))
                return 1;
        }
        if (!cache.commit_append(&error))
            return 1;
    }
    wevision::GraphSession decode(*model, *backend, 512U * 1024U * 1024U);
    wevision::FileTraceSink trace(argv[7],
                                  {"text.decode_hidden", "text.next_logits"});
    decode.set_trace_sink(&trace);
    ggml_tensor *ignored = nullptr;
    ggml_tensor *embedded = nullptr;
    ggml_tensor *hidden = nullptr;
    ggml_tensor *logits = nullptr;
    if (!wevision::build_text_input_embeddings(
            decode, {next_token}, {}, nullptr, &ignored, &embedded, &error))
        return 1;
    std::vector<wevision::TextLayerKV> past;
    for (uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        ggml_tensor *key = ggml_new_tensor_3d(decode.context(), GGML_TYPE_F32,
                                              cache.head_dim(),
                                              cache.kv_heads(), cache.length());
        ggml_tensor *value = ggml_new_tensor_3d(
            decode.context(), GGML_TYPE_F32, cache.head_dim(), cache.kv_heads(),
            cache.length());
        if (!decode.set_input(key, cache.keys(layer), ggml_nbytes(key),
                              &error) ||
            !decode.set_input(value, cache.values(layer), ggml_nbytes(value),
                              &error))
            return 1;
        past.push_back({key, value});
    }
    wevision::TextPositionIds next_positions;
    next_positions.sequence_length = 1;
    next_positions.values.resize(3);
    for (int axis = 0; axis < 3; ++axis)
        next_positions.values[axis] =
            positions
                .values[static_cast<size_t>(axis) * positions.sequence_length +
                        positions.sequence_length - 1] +
            1;
    wevision::TextRotaryData next_rotary;
    std::vector<wevision::TextLayerKV> new_kv;
    if (!wevision::build_text_rotary_data(next_positions, 128, 5000000.0F,
                                          {24, 20, 20}, &next_rotary, &error) ||
        !wevision::build_text_decode(decode, embedded, next_rotary, past,
                                     &new_kv, &hidden, &error) ||
        !wevision::build_text_next_logits(decode, hidden, &logits, &error) ||
        !decode.compute(logits, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
