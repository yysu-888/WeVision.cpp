// HTTP inference service for WeVision. Loads the model once and answers
// POST /infer requests with serialized inference results, plus a GET /health
// endpoint for readiness checks. Inference is serialized with a mutex: the
// resident runtime is not thread-safe, while request parsing, image decoding
// and response encoding run in parallel.
//
// Build:   wevision-serve --model model_gguf/wevision-vl-2b-f16.gguf \
//                          --backend cuda
// Options: --host 0.0.0.0 --port 8717 --max-tokens 256
//
// Protocol (POST /infer, application/json):
//   {"image_path": "images/seabreeze.jpg",     // required, server-side path
//    "task": "detect|point|count|ocr|segment|matting",   // default: detect
//    "query": "a person",                      // used to build default prompts
//    "prompt": "exact prompt",                 // overrides the default prompt
//    "max_tokens": 256, "min_pixels": 16384, "max_pixels": 262144,
//    "mask_threshold": 0.5}
// Response: {"ok":true, "text":..., "count":..., "image_width":...,
//   "boxes":[{"x1":..,"y1":..,"x2":..,"y2":..,"label":..,
//             "mask_png_b64":..., "alpha_png_b64":...}], "points":[...],
//   "timings_ms":{...}} — mask/alpha planes are base64 PNG at the original
//   image resolution, present for segment/matting only.
#include "httplib.h"

#include <nlohmann/json.hpp>

#include "wevision.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

using nlohmann::json;

struct Options {
    std::string model;
    wevision_backend_kind backend = WEVISION_BACKEND_CPU;
    std::string host = "0.0.0.0";
    int port = 8717;
    uint32_t max_tokens = 256;
};

void usage(const char *program) {
    std::fprintf(stderr,
                 "Usage: %s --model PATH [--backend cpu|cuda] "
                 "[--host HOST] [--port N] [--max-tokens N]\n",
                 program);
}

bool parse_options(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const char *value = index + 1 < argc ? argv[index + 1] : nullptr;
        if (flag == "--model" && value) {
            options->model = value;
            ++index;
        } else if (flag == "--backend" && value) {
            const std::string kind = value;
            if (kind == "cpu")
                options->backend = WEVISION_BACKEND_CPU;
            else if (kind == "cuda")
                options->backend = WEVISION_BACKEND_CUDA;
            else
                return false;
            ++index;
        } else if (flag == "--host" && value) {
            options->host = value;
            ++index;
        } else if (flag == "--port" && value) {
            char *end = nullptr;
            options->port = static_cast<int>(std::strtol(value, &end, 10));
            if (value == end || *end != '\0' || options->port <= 0 ||
                options->port > 65535)
                return false;
            ++index;
        } else if (flag == "--max-tokens" && value) {
            char *end = nullptr;
            const long tokens = std::strtol(value, &end, 10);
            if (value == end || *end != '\0' || tokens <= 0 || tokens > 65536)
                return false;
            options->max_tokens = static_cast<uint32_t>(tokens);
            ++index;
        } else {
            return false;
        }
    }
    return !options->model.empty();
}

bool task_from_string(const std::string &name, wevision_task *task) {
    if (name == "detect")
        *task = WEVISION_TASK_DETECT;
    else if (name == "point")
        *task = WEVISION_TASK_POINT;
    else if (name == "count")
        *task = WEVISION_TASK_COUNT;
    else if (name == "ocr")
        *task = WEVISION_TASK_OCR;
    else if (name == "segment")
        *task = WEVISION_TASK_SEGMENT;
    else if (name == "matting")
        *task = WEVISION_TASK_MATTING;
    else
        return false;
    return true;
}

std::string default_prompt(wevision_task task, const std::string &query) {
    const auto phrase = [](const std::string &value) {
        if (value.empty() || value.back() == '.' || value.back() == '?' ||
            value.back() == '!')
            return value;
        return value + ".";
    };
    switch (task) {
    case WEVISION_TASK_DETECT:
        return "Locate all the instances that match the following "
               "description: " +
               query + ".";
    case WEVISION_TASK_POINT:
        return "point to: " + query;
    case WEVISION_TASK_COUNT:
        return "How many " +
               (query.empty() ? std::string("objects") : query) +
               " in the image? Please count them with boxes.";
    case WEVISION_TASK_OCR:
        return "detect all the word in the box format";
    case WEVISION_TASK_SEGMENT:
    case WEVISION_TASK_MATTING:
        return "Locate the instances that match the following description and "
               "provide the alpha mattes: " +
               phrase(query);
    }
    return query;
}

const char *kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const void *data, size_t size) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (size_t index = 0; index < size; index += 3) {
        const uint32_t b0 = bytes[index];
        const uint32_t b1 = index + 1 < size ? bytes[index + 1] : 0;
        const uint32_t b2 = index + 2 < size ? bytes[index + 2] : 0;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kBase64Chars[(triple >> 18) & 0x3F]);
        out.push_back(kBase64Chars[(triple >> 12) & 0x3F]);
        out.push_back(index + 1 < size ? kBase64Chars[(triple >> 6) & 0x3F]
                                       : '=');
        out.push_back(index + 2 < size ? kBase64Chars[triple & 0x3F] : '=');
    }
    return out;
}

struct PngBuffer {
    std::vector<unsigned char> bytes;
};

std::string plane_png_b64(const void *plane, int width, int height) {
    PngBuffer buffer;
    const auto sink = [](void *context, void *data, int size) {
        static_cast<PngBuffer *>(context)->bytes.insert(
            static_cast<PngBuffer *>(context)->bytes.end(),
            static_cast<const unsigned char *>(data),
            static_cast<const unsigned char *>(data) + size);
    };
    if (!stbi_write_png_to_func(sink, &buffer, width, height, 1, plane,
                                width))
        return {};
    return base64_encode(buffer.bytes.data(), buffer.bytes.size());
}

struct ServerState {
    wevision_runtime *runtime = nullptr;
    std::mutex predict_mutex;
    std::atomic<uint64_t> requests_done{0};
    std::atomic<uint64_t> requests_failed{0};
};

std::string alpha_to_b64(const float *alpha, int width, int height) {
    const size_t pixels = static_cast<size_t>(width) * height;
    std::vector<unsigned char> plane(pixels);
    for (size_t pixel = 0; pixel < pixels; ++pixel)
        plane[pixel] = static_cast<unsigned char>(
            std::clamp(alpha[pixel], 0.0F, 1.0F) * 255.0F + 0.5F);
    return plane_png_b64(plane.data(), width, height);
}

void send_error(httplib::Response &response, int status,
                const std::string &message) {
    response.status = status;
    response.set_content(json{{"ok", false}, {"error", message}}.dump(),
                         "application/json");
}

json build_result_json(const wevision_result &result,
                       const wevision_metrics &metrics) {
    json out;
    out["ok"] = true;
    out["text"] = result.text ? result.text : "";
    out["count"] = result.count;
    out["image_width"] = result.image_width;
    out["image_height"] = result.image_height;
    json boxes = json::array();
    for (uint32_t index = 0; index < result.box_count; ++index) {
        const wevision_box_result &box = result.boxes[index];
        json entry;
        entry["x1"] = box.x1;
        entry["y1"] = box.y1;
        entry["x2"] = box.x2;
        entry["y2"] = box.y2;
        entry["label"] = box.label ? box.label : "";
        if (box.mask != nullptr)
            entry["mask_png_b64"] =
                plane_png_b64(box.mask, result.image_width,
                              result.image_height);
        if (box.alpha != nullptr)
            entry["alpha_png_b64"] =
                alpha_to_b64(box.alpha, result.image_width,
                             result.image_height);
        boxes.push_back(std::move(entry));
    }
    out["boxes"] = std::move(boxes);
    json points = json::array();
    for (uint32_t index = 0; index < result.point_count; ++index) {
        const wevision_point_result &point = result.points[index];
        json entry;
        entry["x"] = point.x;
        entry["y"] = point.y;
        entry["label"] = point.label ? point.label : "";
        points.push_back(std::move(entry));
    }
    out["points"] = std::move(points);
    json timings;
    timings["preprocess"] = metrics.preprocess_ms;
    timings["vision"] = metrics.vision_ms;
    timings["prefill"] = metrics.text_prefill_ms;
    timings["decode"] = metrics.text_decode_ms;
    timings["mask"] = metrics.mask_ms;
    timings["tokens"] = metrics.generated_tokens;
    out["timings_ms"] = std::move(timings);
    return out;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }

    const auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };

    const double load_started = now_ms();
    wevision_model *model = nullptr;
    if (wevision_model_load(options.model.c_str(), &model) != WEVISION_OK) {
        std::fprintf(stderr, "Model load failed: %s\n", wevision_last_error());
        return 1;
    }
    const double create_started = now_ms();
    wevision_runtime *runtime = nullptr;
    wevision_runtime_options runtime_options{options.backend, 0,
                                             options.max_tokens};
    const wevision_status status =
        wevision_runtime_create(model, &runtime_options, &runtime);
    wevision_model_free(model);
    if (status != WEVISION_OK) {
        std::fprintf(stderr, "Runtime creation failed: %s\n",
                     wevision_last_error());
        return 1;
    }

    ServerState state;
    state.runtime = runtime;

    httplib::Server server;

    server.Get("/health", [&state, &options](const httplib::Request &,
                                              httplib::Response &response) {
        json out;
        out["status"] = "ok";
        out["backend"] =
            options.backend == WEVISION_BACKEND_CUDA ? "cuda" : "cpu";
        out["model"] = options.model;
        out["requests_done"] = state.requests_done.load();
        out["requests_failed"] = state.requests_failed.load();
        response.set_content(out.dump(), "application/json");
    });

    server.Post("/infer", [&state, &now_ms](const httplib::Request &request,
                                            httplib::Response &response) {
        json body;
        try {
            body = json::parse(request.body);
        } catch (const std::exception &error) {
            state.requests_failed.fetch_add(1);
            send_error(response, 400,
                       std::string("invalid JSON: ") + error.what());
            return;
        }
        if (!body.contains("image_path") || !body["image_path"].is_string() ||
            body["image_path"].get<std::string>().empty()) {
            state.requests_failed.fetch_add(1);
            send_error(response, 400,
                       "missing required string field: image_path");
            return;
        }
        const std::string image_path = body["image_path"];

        wevision_task task = WEVISION_TASK_DETECT;
        if (body.contains("task")) {
            if (!body["task"].is_string() ||
                !task_from_string(body["task"].get<std::string>(), &task)) {
                state.requests_failed.fetch_add(1);
                send_error(response, 400,
                           "invalid task; expected one of "
                           "detect|point|count|ocr|segment|matting");
                return;
            }
        }

        std::string query;
        if (body.contains("query") && body["query"].is_string())
            query = body["query"];
        std::string prompt;
        if (body.contains("prompt") && body["prompt"].is_string() &&
            !body["prompt"].get<std::string>().empty())
            prompt = body["prompt"];
        if (prompt.empty())
            prompt = default_prompt(task, query);

        uint32_t max_tokens = 0;
        uint32_t min_pixels = 0;
        uint32_t max_pixels = 0;
        float mask_threshold = -1.0F;
        const auto number_field = [&body](const char *key, double *out) {
            return body.contains(key) && body[key].is_number() &&
                   (*out = body[key].get<double>(), true);
        };
        double value = 0.0;
        if (number_field("max_tokens", &value) && value >= 1 && value <= 65536)
            max_tokens = static_cast<uint32_t>(value);
        if (number_field("min_pixels", &value) && value >= 0 && value <= 16e6)
            min_pixels = static_cast<uint32_t>(value);
        if (number_field("max_pixels", &value) && value >= 0 && value <= 64e6)
            max_pixels = static_cast<uint32_t>(value);
        if (number_field("mask_threshold", &value) && value >= 0 && value <= 1)
            mask_threshold = static_cast<float>(value);

        int width = 0, height = 0, channels = 0;
        unsigned char *rgb =
            stbi_load(image_path.c_str(), &width, &height, &channels, 3);
        if (rgb == nullptr) {
            state.requests_failed.fetch_add(1);
            send_error(response, 400,
                       std::string("image load failed: ") +
                           stbi_failure_reason());
            return;
        }

        wevision_request inference_request{};
        inference_request.image.rgb = rgb;
        inference_request.image.width = width;
        inference_request.image.height = height;
        inference_request.image.row_stride = width * 3;
        inference_request.prompt = prompt.c_str();
        inference_request.system_prompt = nullptr;
        inference_request.task = task;
        inference_request.max_new_tokens = max_tokens;
        inference_request.vision_min_pixels = min_pixels;
        inference_request.vision_max_pixels = max_pixels;
        inference_request.mask_threshold = mask_threshold;

        wevision_result result{};
        const double started = now_ms();
        {
            // The resident runtime carries mutable decode state; inference is
            // serialized while request parsing and image decoding are not.
            const std::lock_guard<std::mutex> lock(state.predict_mutex);
            if (wevision_predict(state.runtime, &inference_request,
                                 &result) != WEVISION_OK) {
                state.requests_failed.fetch_add(1);
                send_error(response, 500,
                           std::string("inference failed: ") +
                               wevision_runtime_last_error(state.runtime));
                stbi_image_free(rgb);
                return;
            }
        }
        const double elapsed = now_ms();

        wevision_metrics metrics{};
        wevision_runtime_get_last_metrics(state.runtime, &metrics);
        json payload = build_result_json(result, metrics);
        payload["wall_ms"] = elapsed - started;

        wevision_result_free(&result);
        stbi_image_free(rgb);
        state.requests_done.fetch_add(1);
        response.status = 200;
        response.set_content(payload.dump(), "application/json");
    });

    std::fprintf(stderr, "wevision-serve listening on %s:%d (backend=%s)\n",
                 options.host.c_str(), options.port,
                 options.backend == WEVISION_BACKEND_CUDA ? "cuda" : "cpu");
    std::fprintf(stderr, "startup_ms model_load=%.2f runtime_create=%.2f\n",
                 create_started - load_started, now_ms() - create_started);
    if (!server.listen(options.host.c_str(), options.port)) {
        std::fprintf(stderr, "listen failed on %s:%d\n", options.host.c_str(),
                     options.port);
        wevision_runtime_free(runtime);
        return 1;
    }
    wevision_runtime_free(runtime);
    return 0;
}
