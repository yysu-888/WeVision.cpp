// Example CLI: loads a model, runs one prediction, and writes result planes
// and the annotated visualization PNG. Drawing logic lives in visualize.cpp.
#include "visualize.hpp"
#include "wevision.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string image;
    std::string prompt;
    std::string query = "person";
    std::string output_prefix = "wevision-output";
    std::string visualization_path;
    wevision_task task = WEVISION_TASK_SEGMENT;
    wevision_backend_kind backend = WEVISION_BACKEND_CPU;
    uint32_t max_tokens = 256;
    uint32_t min_pixels = 16384;
    uint32_t max_pixels = 16384;
    float threshold = 0.5F;
    uint32_t repeat = 1;
};

void usage(const char *program) {
    std::fprintf(
        stderr,
        "Usage: %s MODEL.gguf IMAGE [options]\n"
        "Options:\n"
        "  --task detect|point|count|ocr|segment|matting\n"
        "  --query TEXT              Object/text query (default: person)\n"
        "  --prompt TEXT             Use an exact model prompt\n"
        "  --backend cpu|metal|cuda   Execution backend (default: cpu)\n"
        "  --max-tokens N            Generation limit (default: 32)\n"
        "  --min-pixels N            Vision canvas lower budget (default: "
        "16384)\n"
        "  --max-pixels N            Vision canvas upper budget (default: "
        "16384)\n"
        "  --threshold F             Mask threshold (default: 0.5)\n"
        "  --output-prefix PATH      Mask/alpha prefix (default: "
        "wevision-output)\n"
        "  --visualize PATH          Write RGB boxes/mask overlay PNG\n"
        "  --repeat N                Load once, predict N times (default: 1);\n"
        "                            reports startup and steady-state timing\n",
        program);
}

bool parse_u32(const char *value, uint32_t *output) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == end || *end != '\0' || parsed > 0xffffffffUL)
        return false;
    *output = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_task(const std::string &value, wevision_task *task) {
    if (value == "detect")
        *task = WEVISION_TASK_DETECT;
    else if (value == "point")
        *task = WEVISION_TASK_POINT;
    else if (value == "count")
        *task = WEVISION_TASK_COUNT;
    else if (value == "ocr")
        *task = WEVISION_TASK_OCR;
    else if (value == "segment")
        *task = WEVISION_TASK_SEGMENT;
    else if (value == "matting")
        *task = WEVISION_TASK_MATTING;
    else
        return false;
    return true;
}

bool parse_options(int argc, char **argv, Options *options) {
    if (argc < 3)
        return false;
    options->model = argv[1];
    options->image = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string flag = argv[index];
        if (index + 1 >= argc)
            return false;
        const char *value = argv[++index];
        if (flag == "--task") {
            if (!parse_task(value, &options->task))
                return false;
        } else if (flag == "--query")
            options->query = value;
        else if (flag == "--prompt")
            options->prompt = value;
        else if (flag == "--output-prefix")
            options->output_prefix = value;
        else if (flag == "--visualize")
            options->visualization_path = value;
        else if (flag == "--backend") {
            if (std::strcmp(value, "cpu") == 0)
                options->backend = WEVISION_BACKEND_CPU;
            else if (std::strcmp(value, "metal") == 0)
                options->backend = WEVISION_BACKEND_METAL;
            else if (std::strcmp(value, "cuda") == 0)
                options->backend = WEVISION_BACKEND_CUDA;
            else
                return false;
        } else if (flag == "--max-tokens") {
            if (!parse_u32(value, &options->max_tokens) ||
                options->max_tokens == 0)
                return false;
        } else if (flag == "--min-pixels") {
            if (!parse_u32(value, &options->min_pixels))
                return false;
        } else if (flag == "--max-pixels") {
            if (!parse_u32(value, &options->max_pixels))
                return false;
        } else if (flag == "--repeat") {
            if (!parse_u32(value, &options->repeat) || options->repeat == 0)
                return false;
        } else if (flag == "--threshold") {
            char *end = nullptr;
            options->threshold = std::strtof(value, &end);
            if (value == end || *end != '\0' || options->threshold < 0.0F ||
                options->threshold > 1.0F)
                return false;
        } else
            return false;
    }
    return options->min_pixels <= options->max_pixels;
}

std::string default_prompt(const Options &options) {
    const auto phrase = [](const std::string &value) {
        if (value.empty() || value.back() == '.' || value.back() == '?' ||
            value.back() == '!')
            return value;
        return value + ".";
    };
    switch (options.task) {
    case WEVISION_TASK_DETECT:
        return "Locate all the instances that match the following "
               "description: " +
               options.query + ".";
    case WEVISION_TASK_POINT:
        return "point to: " + options.query;
    case WEVISION_TASK_COUNT:
        return "How many " +
               (options.query.empty() ? std::string("objects")
                                      : options.query) +
               " in the image? Please count them with boxes.";
    case WEVISION_TASK_OCR:
        return "detect all the word in the box format";
    case WEVISION_TASK_SEGMENT:
    case WEVISION_TASK_MATTING:
        return "Locate the instances that match the following description and "
               "provide the alpha mattes: " +
               phrase(options.query);
    }
    return options.query;
}

bool write_planes(const Options &options, const wevision_result &result) {
    const size_t pixels =
        static_cast<size_t>(result.image_width) * result.image_height;
    std::vector<unsigned char> alpha(pixels);
    for (uint32_t index = 0; index < result.box_count; ++index) {
        const auto &box = result.boxes[index];
        if (box.mask != nullptr) {
            const std::string path = options.output_prefix + "-mask-" +
                                     std::to_string(index) + ".png";
            if (!stbi_write_png(path.c_str(), result.image_width,
                                result.image_height, 1, box.mask,
                                result.image_width))
                return false;
            std::printf("mask[%u]=%s\n", index, path.c_str());
        }
        if (box.alpha != nullptr) {
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                alpha[pixel] = static_cast<unsigned char>(
                    std::clamp(box.alpha[pixel], 0.0F, 1.0F) * 255.0F + 0.5F);
            }
            const std::string path = options.output_prefix + "-alpha-" +
                                     std::to_string(index) + ".png";
            if (!stbi_write_png(path.c_str(), result.image_width,
                                result.image_height, 1, alpha.data(),
                                result.image_width))
                return false;
            std::printf("alpha[%u]=%s\n", index, path.c_str());
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    int width = 0, height = 0, channels = 0;
    unsigned char *rgb =
        stbi_load(options.image.c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr) {
        std::fprintf(stderr, "Image load failed: %s\n", stbi_failure_reason());
        return 1;
    }
    const auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const double load_started = now_ms();
    wevision_model *model = nullptr;
    wevision_status status = wevision_model_load(options.model.c_str(), &model);
    if (status != WEVISION_OK) {
        std::fprintf(stderr, "Model load failed: %s\n", wevision_last_error());
        stbi_image_free(rgb);
        return 1;
    }
    const double create_started = now_ms();
    wevision_runtime_options runtime_options{options.backend, 768,
                                             options.max_tokens};
    wevision_runtime *runtime = nullptr;
    status = wevision_runtime_create(model, &runtime_options, &runtime);
    wevision_model_free(model);
    if (status != WEVISION_OK) {
        std::fprintf(stderr, "Runtime creation failed: %s\n",
                     wevision_last_error());
        stbi_image_free(rgb);
        return 1;
    }
    // runtime_create moves weights onto the backend, so this split shows the
    // resident-service saving: everything here happens once for N predicts.
    std::printf("startup_ms model_load=%.2f runtime_create=%.2f\n",
                create_started - load_started, now_ms() - create_started);
    const std::string prompt =
        options.prompt.empty() ? default_prompt(options) : options.prompt;
    wevision_request request{};
    request.image = {rgb, width, height, width * 3};
    request.prompt = prompt.c_str();
    request.task = options.task;
    request.max_new_tokens = options.max_tokens;
    request.vision_min_pixels = options.min_pixels;
    request.vision_max_pixels = options.max_pixels;
    request.mask_threshold = options.threshold;
    bool written = true;
    bool visualized = true;
    for (uint32_t run = 0; run < options.repeat; ++run) {
        const double predict_started = now_ms();
        wevision_result result{};
        status = wevision_predict(runtime, &request, &result);
        if (status != WEVISION_OK) {
            std::fprintf(stderr, "Prediction failed: %s\n",
                         wevision_runtime_last_error(runtime));
            stbi_image_free(rgb);
            wevision_runtime_free(runtime);
            return 1;
        }
        // Full output only for the first run; later runs keep the files and
        // print timing so steady-state latency is easy to read.
        if (run == 0) {
            std::printf("text=%s\n",
                        result.text == nullptr ? "" : result.text);
            for (uint32_t index = 0; index < result.box_count; ++index) {
                const auto &box = result.boxes[index];
                std::printf("box[%u]=%.2f,%.2f,%.2f,%.2f label=%s\n", index,
                            box.x1, box.y1, box.x2, box.y2,
                            box.label == nullptr ? "" : box.label);
            }
            for (uint32_t index = 0; index < result.point_count; ++index) {
                const auto &point = result.points[index];
                std::printf("point[%u]=%.2f,%.2f label=%s\n", index, point.x,
                            point.y, point.label == nullptr ? "" : point.label);
            }
            if (result.count >= 0)
                std::printf("count=%d\n", result.count);
            written = write_planes(options, result);
            visualized =
                write_visualization(options.visualization_path, result,
                                    options.query, rgb, width, height);
        }
        wevision_metrics metrics{};
        if (wevision_runtime_get_last_metrics(runtime, &metrics) ==
            WEVISION_OK) {
            const double tok_s =
                metrics.text_decode_ms > 0.0
                    ? metrics.generated_tokens * 1000.0 / metrics.text_decode_ms
                    : 0.0;
            std::printf("timing_ms run=%u total=%.2f preprocess=%.2f "
                        "vision=%.2f prefill=%.2f decode=%.2f mask=%.2f "
                        "tokens=%u tok_s=%.2f peak_rss_mb=%.1f\n",
                        run, now_ms() - predict_started, metrics.preprocess_ms,
                        metrics.vision_ms, metrics.text_prefill_ms,
                        metrics.text_decode_ms, metrics.mask_ms,
                        metrics.generated_tokens, tok_s,
                        metrics.peak_rss_bytes / (1024.0 * 1024.0));
            if (metrics.mask_ms > 0.0) {
                std::printf(
                    "mask_stage_ms run=%u encoder=%.2f transformer=%.2f "
                    "msda=%.2f fpn=%.2f pixel_fusion=%.2f coarse=%.2f "
                    "s4=%.2f s2=%.2f s1=%.2f postprocess=%.2f\n",
                    run, metrics.mask_encoder_ms, metrics.mask_transformer_ms,
                    metrics.mask_msda_ms, metrics.mask_fpn_ms,
                    metrics.mask_pixel_fusion_ms, metrics.mask_coarse_ms,
                    metrics.mask_alpha_s4_ms, metrics.mask_alpha_s2_ms,
                    metrics.mask_alpha_s1_ms, metrics.mask_postprocess_ms);
            }
        }
        wevision_result_free(&result);
    }
    stbi_image_free(rgb);
    wevision_runtime_free(runtime);
    if (!written || !visualized) {
        std::fprintf(stderr, "Could not write image output\n");
        return 1;
    }
    return 0;
}
