#include "ggml.h"
#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <vector>

namespace {
bool read(const char *path, std::vector<float> *output) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    input.seekg(0);
    if (bytes < 0 || bytes % 4)
        return false;
    output->resize(static_cast<size_t>(bytes) / 4);
    input.read(reinterpret_cast<char *>(output->data()), bytes);
    return static_cast<bool>(input);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8)
        return 2;
    const bool use_metal = argc == 8 && std::string(argv[7]) == "metal";
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    std::string error;
    auto backend =
        model ? wevision::create_backend_context(
                    *model,
                    use_metal ? WEVISION_BACKEND_METAL : WEVISION_BACKEND_CPU,
                    status, &error)
              : nullptr;
    if (!backend)
        return 1;
    std::vector<float> rgb, pixel, query, coarse;
    if (!read(argv[2], &rgb) || !read(argv[3], &pixel) ||
        !read(argv[4], &query) || !read(argv[5], &coarse) ||
        rgb.size() != 3U * 1024U * 1024U || pixel.size() != 64U * 512U * 512U ||
        query.size() % 192U ||
        coarse.size() != query.size() / 192U * 512U * 512U)
        return 1;
    const int q = query.size() / 192U;
    wevision::GraphSession session(*model, *backend, 512U * 1024U * 1024U);
    std::unique_ptr<wevision::FileTraceSink> trace;
    if (std::string(argv[6]) != "-") {
        trace = std::make_unique<wevision::FileTraceSink>(
            argv[6], std::vector<std::string>{"mask.alpha.*",
                                              "mask.batch.000_001.logits_s1",
                                              "mask.batch.000_001.logits_s2"});
        session.set_trace_sink(trace.get());
    }
    auto *raw_rgb =
        ggml_new_tensor_3d(session.context(), GGML_TYPE_F32, 1024, 1024, 3);
    auto *raw_pixel =
        ggml_new_tensor_3d(session.context(), GGML_TYPE_F32, 512, 512, 64);
    auto *queries =
        ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 192, q);
    auto *coarse_logits =
        ggml_new_tensor_3d(session.context(), GGML_TYPE_F32, 512, 512, q);
    if (!session.set_input(raw_rgb, rgb.data(), ggml_nbytes(raw_rgb), &error) ||
        !session.set_input(raw_pixel, pixel.data(), ggml_nbytes(raw_pixel),
                           &error) ||
        !session.set_input(queries, query.data(), ggml_nbytes(queries),
                           &error) ||
        !session.set_input(coarse_logits, coarse.data(),
                           ggml_nbytes(coarse_logits), &error))
        return 1;
    auto *image =
        ggml_cont(session.context(),
                  ggml_permute(session.context(), raw_rgb, 1, 2, 0, 3));
    auto *feature =
        ggml_cont(session.context(),
                  ggml_permute(session.context(), raw_pixel, 1, 2, 0, 3));
    const bool profile = std::getenv("WEVISION_ALPHA_PROFILE") != nullptr;
    std::vector<wevision::GraphProfilePoint> profile_points;
    std::fprintf(stderr, "build alpha\n");
    wevision::MaskAlphaOutput output;
    if (!wevision::build_mask_alpha_cascade(
            session, image, feature, queries, coarse_logits, &output, &error,
            profile ? &profile_points : nullptr)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "build trace\n");
    if (!session.checkpoint(
            {"mask.batch.000_001.logits_s1", "mask", "objects,H,W"},
            output.logits_s1, &error) ||
        !session.checkpoint(
            {"mask.batch.000_001.logits_s2", "mask", "objects,H,W"},
            output.logits_s2, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "compute alpha\n");
    const auto started = std::chrono::steady_clock::now();
    if (profile) {
        std::vector<wevision::GraphProfileSample> samples;
        if (!session.compute_profiled(output.logits_s1, profile_points,
                                      &samples, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (const auto &sample : samples)
            std::fprintf(stderr, "alpha_profile %s %.2f\n", sample.name.c_str(),
                         sample.milliseconds);
    } else if (!session.compute(output.logits_s1, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const auto finished = std::chrono::steady_clock::now();
    std::fprintf(
        stderr, "alpha_compute_ms=%.2f\n",
        std::chrono::duration<double, std::milli>(finished - started).count());
    return 0;
}
