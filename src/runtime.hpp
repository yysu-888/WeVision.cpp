#ifndef WEVISION_RUNTIME_HPP
#define WEVISION_RUNTIME_HPP

#include "model_loader.hpp"
#include "trace.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_cgraph;
struct gguf_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;

typedef struct ggml_backend *ggml_backend_t;
typedef struct ggml_backend_buffer *ggml_backend_buffer_t;
typedef struct ggml_gallocr *ggml_gallocr_t;

namespace wevision {

inline void set_error(std::string *destination, std::string value) {
    if (destination != nullptr)
        *destination = std::move(value);
}

// Single source of truth for the backend kinds this library recognizes.
// wevision_runtime_create's argument validation and create_backend_context's
// dispatch both consult this list; adding a backend means adding one case
// here and one in the switch.
inline bool is_supported_backend(wevision_backend_kind kind) {
    switch (kind) {
    case WEVISION_BACKEND_CPU:
    case WEVISION_BACKEND_METAL:
    case WEVISION_BACKEND_CUDA:
        return true;
    default:
        return false;
    }
}

struct GraphProfilePoint {
    std::string name;
    ggml_tensor *tensor = nullptr;
    // Opaque caller-defined marker, mirrored into GraphProfileSample. The
    // session never interprets it; callers with structured dispatch match on
    // it instead of parsing `name`.
    int tag = 0;
};

struct GraphProfileSample {
    std::string name;
    double milliseconds = 0.0;
    int tag = 0;
};

// Device-bound realization of one immutable Model. It owns a separate GGUF
// descriptor context so multiple CPU/Metal instances never share mutable
// ggml tensor buffer assignments.
class BackendContext {
  public:
    BackendContext() = default;
    ~BackendContext();

    BackendContext(const BackendContext &) = delete;
    BackendContext &operator=(const BackendContext &) = delete;

    bool ready() const { return backend_ != nullptr && weights_ != nullptr; }

    wevision_backend_kind kind() const { return kind_; }

    // Metal's mask refinement uses an N==1 lowered conv2d fast path that avoids
    // permute+cont copies, so per-object batching is faster than N>1 batching.
    bool prefers_single_object_mask_batch() const {
        return kind_ == WEVISION_BACKEND_METAL;
    }

    // CPU BF16 reductions accumulate error across deep decoder stacks, whereas
    // GPU backends execute BF16 matmuls natively without per-step F32
    // expansion.
    bool promotes_bf16_matmul_to_f32() const {
        return kind_ == WEVISION_BACKEND_CPU;
    }

    ggml_backend_t backend() const { return backend_; }

    ggml_tensor *tensor(const std::string &name, std::string *error) const;

  private:
    friend std::unique_ptr<BackendContext>
    create_backend_context(const Model &, wevision_backend_kind,
                           wevision_status &, std::string *);

    wevision_backend_kind kind_ = WEVISION_BACKEND_CPU;
    ::gguf_context *gguf_ = nullptr;
    ::ggml_context *metadata_ = nullptr;
    std::unordered_map<std::string, ::ggml_tensor *> tensors_;
    ::ggml_backend_t backend_ = nullptr;
    ::ggml_backend_buffer_t weights_ = nullptr;
};

std::unique_ptr<BackendContext>
create_backend_context(const Model &model, wevision_backend_kind kind,
                       wevision_status &status, std::string *error);

// Owns temporary graph tensors for one inference request. Model weights are
// borrowed from BackendContext and never copied into this context.
class GraphSession {
  public:
    GraphSession(const Model &model, const BackendContext &backend,
                 size_t graph_memory_bytes = 16U * 1024U * 1024U,
                 ggml_gallocr_t reusable_allocator = nullptr);
    ~GraphSession();

    GraphSession(const GraphSession &) = delete;
    GraphSession &operator=(const GraphSession &) = delete;

    bool ready(std::string *error) const;
    ggml_tensor *weight(const std::string &name, std::string *error) const;

    const Model &model() const { return model_; }

    wevision_backend_kind backend_kind() const { return backend_.kind(); }

    bool promotes_bf16_matmul_to_f32() const {
        return backend_.promotes_bf16_matmul_to_f32();
    }

    // Binds an optional build-time hook. It is intentionally internal: the
    // public C API remains free of tracing and debug-lifetime concerns.
    void set_trace_sink(TraceSink *trace) { trace_ = trace; }

    // True while a trace sink is bound. Graph builders may skip building
    // trace-only tensors (layout-export copies) when this returns false; the
    // underlying graph result is identical either way.
    bool trace_attached() const { return trace_ != nullptr; }

    bool checkpoint(TraceTensorInfo info, ggml_tensor *tensor,
                    std::string *error);

    // Allocates all temporary tensors reachable from output and executes its
    // forward graph. Output data remains valid until reset() or destruction.
    bool compute(ggml_tensor *output, std::string *error,
                 TraceSink *trace = nullptr);
    bool compute(const std::vector<ggml_tensor *> &outputs, std::string *error,
                 TraceSink *trace = nullptr);
    bool compute_profiled(ggml_tensor *output,
                          const std::vector<GraphProfilePoint> &points,
                          std::vector<GraphProfileSample> *samples,
                          std::string *error);
    bool compute_profiled(const std::vector<ggml_tensor *> &outputs,
                          const std::vector<GraphProfilePoint> &points,
                          std::vector<GraphProfileSample> *samples,
                          std::string *error);
    bool set_input(ggml_tensor *tensor, const void *data, size_t bytes,
                   std::string *error);
    bool copy_to_host(const ggml_tensor *tensor, void *data, size_t bytes,
                      std::string *error) const;
    bool copy_to_host_range(const ggml_tensor *tensor, void *data,
                            size_t offset, size_t bytes,
                            std::string *error) const;
    // Retains small host-side descriptors referenced by ggml custom-op
    // callbacks until this graph session is destroyed or reset.
    void *allocate_custom_data(size_t bytes);
    void reset();

    ggml_context *context() const { return context_; }

  private:
    void create_context();
    bool compute_impl(const std::vector<ggml_tensor *> &outputs,
                      std::string *error, TraceSink *trace,
                      const std::vector<GraphProfilePoint> *profile_points,
                      std::vector<GraphProfileSample> *profile_samples);
    // compute_impl's execution strategies: per-operation profiling,
    // profile-point sampling, and the plain backend run stays inline.
    // Validation failures are reported via `error` plus reset(); the final
    // backend status is stored through `graph_status` (int-valued
    // enum ggml_status) so compute_impl can run the shared post-compute
    // failure handling.
    bool execute_operation_profiled(
        ggml_cgraph *graph,
        const std::vector<GraphProfilePoint> *profile_points,
        std::vector<GraphProfileSample> *profile_samples, int *graph_status,
        std::string *error);
    bool execute_profiled(ggml_cgraph *graph,
                          const std::vector<GraphProfilePoint> *profile_points,
                          std::vector<GraphProfileSample> *profile_samples,
                          int *graph_status, std::string *error);

    const Model &model_;
    const BackendContext &backend_;
    size_t graph_memory_bytes_ = 0;
    ggml_context *context_ = nullptr;
    // Reuses activation storage by graph lifetime, which is essential for
    // full-resolution mask refinement.
    ggml_gallocr_t allocator_ = nullptr;
    ggml_gallocr_t reusable_allocator_ = nullptr;

    struct PendingInput {
        ggml_tensor *tensor = nullptr;
        std::vector<uint8_t> data;
    };

    std::vector<PendingInput> pending_inputs_;

    struct Checkpoint {
        TraceTensorInfo info;
        ggml_tensor *tensor = nullptr;
    };

    TraceSink *trace_ = nullptr;
    std::vector<Checkpoint> checkpoints_;
    std::vector<std::unique_ptr<uint8_t[]>> custom_data_;
};

} // namespace wevision

#endif
