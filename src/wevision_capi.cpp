#include "inference.hpp"
#include "runtime.hpp"
#include "tokenizer.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <utility>

struct wevision_model {
    std::shared_ptr<wevision::Model> impl;
};

struct wevision_runtime {
    std::shared_ptr<wevision::Model> model;
    std::unique_ptr<wevision::BackendContext> backend;
    wevision::QwenTokenizer tokenizer;

    size_t graph_memory_bytes = 32U * 1024U * 1024U;
    uint32_t max_new_tokens = 256;
    wevision_metrics last_metrics{};
    std::string last_error;
};

namespace {
thread_local std::string g_last_error;

// Keeps the thread-global diagnostic in sync with the per-runtime one so both
// wevision_last_error() and wevision_runtime_last_error() report the failure.
void set_runtime_error(wevision_runtime &runtime, std::string message) {
    runtime.last_error = std::move(message);
    g_last_error = runtime.last_error;
}

char *duplicate_string(const std::string &value) {
    auto *result = static_cast<char *>(std::malloc(value.size() + 1));
    if (result == nullptr)
        return nullptr;
    std::memcpy(result, value.c_str(), value.size() + 1);
    return result;
}
} // namespace

extern "C" wevision_status wevision_model_load(const char *path,
                                               wevision_model **out_model) {
    if (path == nullptr || out_model == nullptr)
        return WEVISION_ERROR_INVALID_ARGUMENT;
    *out_model = nullptr;
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::unique_ptr<wevision::Model> loaded =
        wevision::load_model(path, status);
    if (loaded == nullptr) {
        g_last_error = wevision_status_string(status);
        return status;
    }
    // A partially-loaded model (metadata OK, validation failed) is not
    // observable through the C API; report the diagnostic and free it.
    if (status != WEVISION_OK) {
        g_last_error = loaded->last_error;
        return status;
    }
    auto handle = std::make_unique<wevision_model>();
    handle->impl = std::move(loaded);
    g_last_error.clear();
    *out_model = handle.release();
    return WEVISION_OK;
}

extern "C" void wevision_model_free(wevision_model *model) { delete model; }

extern "C" wevision_status
wevision_runtime_create(const wevision_model *model,
                        const wevision_runtime_options *options,
                        wevision_runtime **out_runtime) {
    if (model == nullptr || out_runtime == nullptr) {
        return WEVISION_ERROR_INVALID_ARGUMENT;
    }
    *out_runtime = nullptr;
    const wevision_backend_kind backend =
        options == nullptr ? WEVISION_BACKEND_CPU : options->backend;
    if (!wevision::is_supported_backend(backend)) {
        g_last_error = "unknown runtime backend";
        return WEVISION_ERROR_INVALID_ARGUMENT;
    }
    auto runtime = std::make_unique<wevision_runtime>();
    runtime->model = model->impl;
    if (options != nullptr) {
        if (options->graph_memory_megabytes != 0) {
            runtime->graph_memory_bytes =
                static_cast<size_t>(options->graph_memory_megabytes) * 1024U *
                1024U;
        }
        if (options->max_new_tokens != 0) {
            runtime->max_new_tokens = options->max_new_tokens;
        }
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    runtime->backend = wevision::create_backend_context(
        *runtime->model, backend, status, &runtime->last_error);
    if (!runtime->backend) {
        g_last_error = runtime->last_error;
        return status;
    }
    if (!runtime->tokenizer.load(runtime->model->tokenizer,
                                 &runtime->last_error)) {
        g_last_error = runtime->last_error;
        return WEVISION_ERROR_MODEL_FORMAT;
    }
    g_last_error.clear();
    *out_runtime = runtime.release();
    return WEVISION_OK;
}

extern "C" void wevision_runtime_free(wevision_runtime *runtime) {
    delete runtime;
}

extern "C" const char *
wevision_runtime_last_error(const wevision_runtime *runtime) {
    return runtime == nullptr ? "invalid runtime handle"
                              : runtime->last_error.c_str();
}

extern "C" wevision_status
wevision_runtime_get_last_metrics(const wevision_runtime *runtime,
                                  wevision_metrics *out_metrics) {
    if (runtime == nullptr || out_metrics == nullptr)
        return WEVISION_ERROR_INVALID_ARGUMENT;
    *out_metrics = runtime->last_metrics;
    return WEVISION_OK;
}

extern "C" void wevision_result_free(wevision_result *result) {
    if (result == nullptr)
        return;
    std::free(result->text);
    if (result->boxes != nullptr) {
        for (uint32_t index = 0; index < result->box_count; ++index) {
            std::free(result->boxes[index].label);
            std::free(result->boxes[index].mask);
            std::free(result->boxes[index].alpha);
        }
    }
    if (result->points != nullptr) {
        for (uint32_t index = 0; index < result->point_count; ++index)
            std::free(result->points[index].label);
    }
    std::free(result->boxes);
    std::free(result->points);
    *result = {};
    result->count = -1;
}

extern "C" wevision_status wevision_predict(wevision_runtime *runtime,
                                            const wevision_request *request,
                                            wevision_result *out_result) {
    if (runtime == nullptr || request == nullptr || out_result == nullptr)
        return WEVISION_ERROR_INVALID_ARGUMENT;
    *out_result = {};
    out_result->count = -1;
    try {
        wevision::InferenceOutput inference;
        if (!wevision::run_inference(
                *runtime->model, *runtime->backend, runtime->tokenizer,
                runtime->graph_memory_bytes, runtime->max_new_tokens, *request,
                &inference, &runtime->last_error)) {
            g_last_error = runtime->last_error;
            return WEVISION_ERROR_INTERNAL;
        }
        runtime->last_metrics = inference.metrics;

        struct rusage usage {};

        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
            runtime->last_metrics.peak_rss_bytes =
                static_cast<uint64_t>(usage.ru_maxrss);
#else
            runtime->last_metrics.peak_rss_bytes =
                static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
#endif
        }
        out_result->text = duplicate_string(inference.text);
        out_result->box_count =
            static_cast<uint32_t>(inference.parsed.boxes.size());
        out_result->point_count =
            static_cast<uint32_t>(inference.parsed.points.size());
        out_result->count = inference.parsed.count;
        out_result->image_width = inference.image_width;
        out_result->image_height = inference.image_height;
        if (out_result->box_count != 0) {
            out_result->boxes = static_cast<wevision_box_result *>(std::calloc(
                out_result->box_count, sizeof(wevision_box_result)));
        }
        if (out_result->point_count != 0) {
            out_result->points =
                static_cast<wevision_point_result *>(std::calloc(
                    out_result->point_count, sizeof(wevision_point_result)));
        }
        if (out_result->text == nullptr ||
            (out_result->box_count != 0 && out_result->boxes == nullptr) ||
            (out_result->point_count != 0 && out_result->points == nullptr)) {
            wevision_result_free(out_result);
            set_runtime_error(*runtime, "cannot allocate inference result");
            return WEVISION_ERROR_OUT_OF_MEMORY;
        }
        const size_t plane =
            static_cast<size_t>(inference.image_width) * inference.image_height;
        const size_t mask_objects =
            plane == 0 ? 0 : inference.alpha.size() / plane;
        for (uint32_t index = 0; index < out_result->box_count; ++index) {
            const auto &source = inference.parsed.boxes[index];
            auto &destination = out_result->boxes[index];
            destination.x1 = source.x1;
            destination.y1 = source.y1;
            destination.x2 = source.x2;
            destination.y2 = source.y2;
            destination.label = duplicate_string(source.label);
            if (destination.label == nullptr) {
                wevision_result_free(out_result);
                set_runtime_error(*runtime, "cannot allocate box label");
                return WEVISION_ERROR_OUT_OF_MEMORY;
            }
            if (index < mask_objects) {
                destination.mask = static_cast<uint8_t *>(std::malloc(plane));
                destination.alpha =
                    static_cast<float *>(std::malloc(plane * sizeof(float)));
                if (destination.mask == nullptr ||
                    destination.alpha == nullptr) {
                    wevision_result_free(out_result);
                    set_runtime_error(*runtime, "cannot allocate mask result");
                    return WEVISION_ERROR_OUT_OF_MEMORY;
                }
                std::memcpy(destination.mask,
                            inference.mask.data() +
                                static_cast<size_t>(index) * plane,
                            plane);
                std::memcpy(destination.alpha,
                            inference.alpha.data() +
                                static_cast<size_t>(index) * plane,
                            plane * sizeof(float));
            }
        }
        for (uint32_t index = 0; index < out_result->point_count; ++index) {
            const auto &source = inference.parsed.points[index];
            out_result->points[index].x = source.x;
            out_result->points[index].y = source.y;
            out_result->points[index].label = duplicate_string(source.label);
            if (out_result->points[index].label == nullptr) {
                wevision_result_free(out_result);
                set_runtime_error(*runtime, "cannot allocate point label");
                return WEVISION_ERROR_OUT_OF_MEMORY;
            }
        }
        runtime->last_error.clear();
        g_last_error.clear();
        return WEVISION_OK;
    } catch (const std::bad_alloc &) {
        wevision_result_free(out_result);
        set_runtime_error(*runtime, "out of memory during inference");
        return WEVISION_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        wevision_result_free(out_result);
        set_runtime_error(*runtime, "unexpected inference failure");
        return WEVISION_ERROR_INTERNAL;
    }
}

extern "C" const char *wevision_status_string(wevision_status status) {
    switch (status) {
    case WEVISION_OK:
        return "ok";
    case WEVISION_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case WEVISION_ERROR_FILE_NOT_FOUND:
        return "file not found";
    case WEVISION_ERROR_MODEL_FORMAT:
        return "invalid model format";
    case WEVISION_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case WEVISION_ERROR_INTERNAL:
        return "internal error";
    }
    return "unknown status";
}

extern "C" const char *wevision_last_error(void) {
    return g_last_error.c_str();
}
