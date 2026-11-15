#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include "vision.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4 || argc > 8 || argc == 6) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TRACE_DIR PATCHES.f32 [LAST_LAYER "
                     "[GRID_H GRID_W [cpu|metal]]]\n",
                     argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    std::string error;
    const auto backend_kind = argc == 8 && std::string(argv[7]) == "metal"
                                  ? WEVISION_BACKEND_METAL
                                  : WEVISION_BACKEND_CPU;
    auto backend = model == nullptr ? nullptr
                                    : wevision::create_backend_context(
                                          *model, backend_kind, status, &error);
    if (backend == nullptr) {
        std::fprintf(stderr, "Backend: %s\n", error.c_str());
        return 1;
    }
    wevision::PatchifiedImage patches;
    patches.grid_h = argc >= 7 ? std::atoi(argv[5]) : 6;
    patches.grid_w = argc >= 7 ? std::atoi(argv[6]) : 8;
    patches.patch_dim = 1536;
    if (patches.grid_h <= 0 || patches.grid_w <= 0)
        return 2;
    patches.values.resize(static_cast<size_t>(patches.grid_h) * patches.grid_w *
                          patches.patch_dim);
    std::ifstream input(argv[3], std::ios::binary);
    input.read(
        reinterpret_cast<char *>(patches.values.data()),
        static_cast<std::streamsize>(patches.values.size() * sizeof(float)));
    if (!input) {
        std::fputs("Invalid patch payload\n", stderr);
        return 1;
    }
    const int last_layer = argc >= 5 ? std::atoi(argv[4]) : 5;
    if (last_layer < 0 || last_layer >= 24) {
        std::fputs("LAST_LAYER must be 0..23\n", stderr);
        return 2;
    }
    wevision::GraphSession session(*model, *backend, 768U * 1024U * 1024U);
    std::unique_ptr<wevision::FileTraceSink> hook_trace;
    if (last_layer == 23 && std::string(argv[2]) != "-") {
        hook_trace = std::make_unique<wevision::FileTraceSink>(
            argv[2], std::vector<std::string>{"vision.deepstack.*",
                                              "vision.merger.output"});
        session.set_trace_sink(hook_trace.get());
    }
    ggml_tensor *hidden = nullptr;
    if (!wevision::build_vision_patch_embed(session, patches, &hidden,
                                            &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    wevision::VisionPositionData positions;
    wevision::VisionRotaryData rotary;
    if (!wevision::build_vision_position_data(patches.grid_h, patches.grid_w, 2,
                                              48, &positions) ||
        !wevision::build_vision_positioned_input(session, hidden, positions,
                                                 &hidden, &error) ||
        !wevision::build_vision_rotary_data(positions, 64, 10000.0F, &rotary)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::array<ggml_tensor *, 3> deepstack_outputs{};
    for (int layer = 0; layer <= last_layer; ++layer) {
        if (!wevision::build_vision_block(session, layer, hidden, rotary,
                                          &hidden, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const int deepstack_index = layer == 5    ? 0
                                    : layer == 11 ? 1
                                    : layer == 17 ? 2
                                                  : -1;
        if (deepstack_index >= 0 &&
            !wevision::build_vision_merger(
                session, hidden,
                "wv.vision.deepstack." + std::to_string(deepstack_index), true,
                &deepstack_outputs[static_cast<size_t>(deepstack_index)],
                &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    }
    if (last_layer == 23 &&
        !wevision::build_vision_merger(session, hidden, "wv.vision.merger",
                                       false, &hidden, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (last_layer == 23) {
        const auto started = std::chrono::steady_clock::now();
        if (!session.compute(hidden, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const auto finished = std::chrono::steady_clock::now();
        std::fprintf(
            stderr, "vision_compute_ms=%.2f\n",
            std::chrono::duration<double, std::milli>(finished - started)
                .count());
        return 0;
    }
    wevision::FileTraceSink trace(argv[2]);
    const std::string name =
        last_layer == 23
            ? "vision.merger.output"
            : std::string("vision.block.") + (last_layer < 10 ? "0" : "") +
                  std::to_string(last_layer) + ".premerge_tokens";
    if (!trace.watch({name, "vision", "patch,hidden"}, hidden, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *completion = hidden;
    for (size_t index = 0; index < deepstack_outputs.size(); ++index) {
        if (deepstack_outputs[index] == nullptr)
            continue;
        const std::string deepstack_name =
            "vision.deepstack." + std::to_string(index) + ".output";
        if (!trace.watch({deepstack_name, "vision", "merged_patch,text_hidden"},
                         deepstack_outputs[index], &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        // Keep independently-produced deepstack tensors reachable in this
        // test-only graph.
        completion =
            ggml_add(session.context(), completion, deepstack_outputs[index]);
    }
    if (!session.compute(completion, &error, &trace)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
