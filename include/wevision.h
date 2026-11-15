#ifndef WEVISION_H
#define WEVISION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wevision_model wevision_model;
typedef struct wevision_runtime wevision_runtime;

typedef enum wevision_status {
    WEVISION_OK = 0,
    WEVISION_ERROR_INVALID_ARGUMENT = 1,
    WEVISION_ERROR_FILE_NOT_FOUND = 2,
    WEVISION_ERROR_MODEL_FORMAT = 3,
    WEVISION_ERROR_OUT_OF_MEMORY = 4,
    WEVISION_ERROR_INTERNAL = 5,
} wevision_status;

typedef enum wevision_backend_kind {
    WEVISION_BACKEND_CPU = 0,
    WEVISION_BACKEND_METAL = 1,
    WEVISION_BACKEND_CUDA = 2,
} wevision_backend_kind;

typedef struct wevision_runtime_options {
    wevision_backend_kind backend;
    // Zero selects the implementation default.
    uint32_t graph_memory_megabytes;
    // Zero selects the implementation default (128).
    uint32_t max_new_tokens;
} wevision_runtime_options;

typedef enum wevision_task {
    WEVISION_TASK_DETECT = 0,
    WEVISION_TASK_POINT = 1,
    WEVISION_TASK_COUNT = 2,
    WEVISION_TASK_OCR = 3,
    WEVISION_TASK_SEGMENT = 4,
    WEVISION_TASK_MATTING = 5,
} wevision_task;

typedef struct wevision_image_view {
    const uint8_t *rgb;
    int32_t width;
    int32_t height;
    // Bytes between adjacent rows. Zero means width * 3.
    int32_t row_stride;
} wevision_image_view;

typedef struct wevision_request {
    wevision_image_view image;
    const char *prompt;
    const char *system_prompt;
    wevision_task task;
    // Zero uses the runtime default.
    uint32_t max_new_tokens;
    // Zero uses the processor defaults. Tests and latency-sensitive callers
    // may pin both values to the same canvas budget.
    uint32_t vision_min_pixels;
    uint32_t vision_max_pixels;
    // Values outside [0, 1] use 0.5.
    float mask_threshold;
} wevision_request;

typedef struct wevision_box_result {
    float x1;
    float y1;
    float x2;
    float y2;
    char *label;
    // Row-major original-image dimensions. Present for segmentation/matting.
    uint8_t *mask;
    float *alpha;
} wevision_box_result;

typedef struct wevision_point_result {
    float x;
    float y;
    char *label;
} wevision_point_result;

typedef struct wevision_result {
    char *text;
    wevision_box_result *boxes;
    uint32_t box_count;

    wevision_point_result *points;
    uint32_t point_count;

    int32_t count;
    int32_t image_width;
    int32_t image_height;
} wevision_result;

typedef struct wevision_metrics {
    double preprocess_ms;
    double vision_ms;
    double text_prefill_ms;
    double text_decode_ms;
    double mask_ms;
    double mask_encoder_ms;
    double mask_transformer_ms;
    double mask_msda_ms;
    double mask_fpn_ms;
    double mask_pixel_fusion_ms;
    double mask_coarse_ms;
    double mask_alpha_s4_ms;
    double mask_alpha_s2_ms;
    double mask_alpha_s1_ms;
    double mask_postprocess_ms;
    uint32_t generated_tokens;
    uint64_t peak_rss_bytes;
} wevision_metrics;

// Loads GGUF metadata and tensor descriptors only. It does not allocate model
// weights on a CPU/GPU backend and does not run inference. On failure
// *out_model is nullptr and wevision_last_error() carries the diagnostic.
wevision_status wevision_model_load(const char *path,
                                    wevision_model **out_model);

void wevision_model_free(wevision_model *model);

// Creates an independent execution context and uploads the model weights to
// the selected backend. The runtime retains the immutable model data, so the
// model handle may be freed after this call succeeds.
wevision_status wevision_runtime_create(const wevision_model *model,
                                        const wevision_runtime_options *options,
                                        wevision_runtime **out_runtime);

void wevision_runtime_free(wevision_runtime *runtime);

const char *wevision_runtime_last_error(const wevision_runtime *runtime);

wevision_status
wevision_runtime_get_last_metrics(const wevision_runtime *runtime,
                                  wevision_metrics *out_metrics);

wevision_status wevision_predict(wevision_runtime *runtime,
                                 const wevision_request *request,
                                 wevision_result *out_result);

void wevision_result_free(wevision_result *result);

const char *wevision_status_string(wevision_status status);
// Returns the diagnostic from the most recent failed call on this thread.
const char *wevision_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
