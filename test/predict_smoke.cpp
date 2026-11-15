#include "wevision.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
bool token(FILE *file, std::string *value) {
    value->clear();
    int c = 0;
    do {
        c = std::fgetc(file);
    } while (c == ' ' || c == '\n' || c == '\r' || c == '\t');
    while (c != EOF && c != ' ' && c != '\n' && c != '\r' && c != '\t') {
        value->push_back(static_cast<char>(c));
        c = std::fgetc(file);
    }
    return !value->empty();
}

bool read_ppm(const char *path, int *width, int *height,
              std::vector<unsigned char> *rgb) {
    FILE *file = std::fopen(path, "rb");
    if (!file)
        return false;
    std::string magic, w, h, max;
    const bool ok = token(file, &magic) && token(file, &w) && token(file, &h) &&
                    token(file, &max);
    if (!ok || magic != "P6" || max != "255") {
        std::fclose(file);
        return false;
    }
    *width = std::atoi(w.c_str());
    *height = std::atoi(h.c_str());
    rgb->resize(static_cast<size_t>(*width) * *height * 3U);
    const bool read =
        std::fread(rgb->data(), 1, rgb->size(), file) == rgb->size();
    std::fclose(file);
    return read;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;
    if (!read_ppm(argv[2], &width, &height, &rgb))
        return 1;
    wevision_model *model = nullptr;
    if (wevision_model_load(argv[1], &model) != WEVISION_OK)
        return 1;
    wevision_runtime_options options{WEVISION_BACKEND_CPU, 768, 32};
    wevision_runtime *runtime = nullptr;
    if (wevision_runtime_create(model, &options, &runtime) != WEVISION_OK)
        return 1;
    wevision_model_free(model);
    wevision_request request{};
    request.image = {rgb.data(), width, height, width * 3};
    request.prompt =
        "Locate the instances that match the following description "
        "and provide the alpha mattes: person.";
    request.task = WEVISION_TASK_SEGMENT;
    request.max_new_tokens = 32;
    request.vision_min_pixels = 16384;
    request.vision_max_pixels = 16384;
    request.mask_threshold = 0.5F;
    wevision_result result{};
    const wevision_status status = wevision_predict(runtime, &request, &result);
    if (status != WEVISION_OK) {
        std::fprintf(stderr, "%s\n", wevision_runtime_last_error(runtime));
        wevision_runtime_free(runtime);
        return 1;
    }
    std::printf("text=%s\nboxes=%u points=%u count=%d\n", result.text,
                result.box_count, result.point_count, result.count);
    wevision_metrics metrics{};
    if (wevision_runtime_get_last_metrics(runtime, &metrics) == WEVISION_OK) {
        const double decode_seconds = metrics.text_decode_ms / 1000.0;
        const double tokens_per_second =
            decode_seconds > 0.0 ? metrics.generated_tokens / decode_seconds
                                 : 0.0;
        std::printf(
            "preprocess_ms=%.2f vision_ms=%.2f prefill_ms=%.2f "
            "decode_ms=%.2f mask_ms=%.2f tokens=%u tok_s=%.2f "
            "peak_rss=%llu\n",
            metrics.preprocess_ms, metrics.vision_ms, metrics.text_prefill_ms,
            metrics.text_decode_ms, metrics.mask_ms, metrics.generated_tokens,
            tokens_per_second, (unsigned long long)metrics.peak_rss_bytes);
    }
    const bool valid = result.text != nullptr && result.box_count > 0 &&
                       result.boxes[0].mask != nullptr &&
                       result.boxes[0].alpha != nullptr;
    wevision_result_free(&result);
    wevision_runtime_free(runtime);
    return valid ? 0 : 1;
}
