#include "runtime.hpp"

#include "profile.hpp"
#include "trace.hpp"

// Needed for ggml_cgraph struct definition in make_graph_view().
#include "../third_party/ggml/src/ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#if defined(WEVISION_USE_METAL)
#include "ggml-metal.h"
#endif

#if defined(WEVISION_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wevision {

namespace {

ggml_cgraph make_graph_view(ggml_cgraph *graph, int begin, int end) {
    ggml_cgraph view{};
    view.n_nodes = end - begin;
    view.nodes = graph->nodes + begin;
    view.use_counts = graph->use_counts;
    view.visited_hash_set = graph->visited_hash_set;
    view.order = graph->order;
    return view;
}

bool operation_profile_enabled() {
    return DebugOptions::from_environment().operation_profile;
}

int operation_profile_index(enum ggml_op op) {
    switch (op) {
    case GGML_OP_PERMUTE:
        return 0;
    case GGML_OP_CONT:
        return 1;
    case GGML_OP_REPEAT:
        return 2;
    case GGML_OP_CONCAT:
        return 3;
    case GGML_OP_IM2COL:
        return 4;
    case GGML_OP_MUL_MAT:
        return 5;
    default:
        return -1;
    }
}

void print_profile_tensor(const char *label, const ggml_tensor *tensor) {
    if (tensor == nullptr)
        return;
    std::fprintf(stderr, " %s=%s[%lld,%lld,%lld,%lld]/%zuB", label,
                 ggml_type_name(tensor->type),
                 static_cast<long long>(tensor->ne[0]),
                 static_cast<long long>(tensor->ne[1]),
                 static_cast<long long>(tensor->ne[2]),
                 static_cast<long long>(tensor->ne[3]), ggml_nbytes(tensor));
}

} // namespace

BackendContext::~BackendContext() {
    if (weights_ != nullptr)
        ggml_backend_buffer_free(weights_);
    if (backend_ != nullptr)
        ggml_backend_free(backend_);
    if (gguf_ != nullptr)
        gguf_free(gguf_);
    if (metadata_ != nullptr)
        ggml_free(metadata_);
}

ggml_tensor *BackendContext::tensor(const std::string &name,
                                    std::string *error) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        set_error(error, "unknown backend tensor: " + name);
        return nullptr;
    }
    return found->second;
}

std::unique_ptr<BackendContext>
create_backend_context(const Model &model, wevision_backend_kind kind,
                       wevision_status &status, std::string *error) {
    status = WEVISION_ERROR_INTERNAL;
    if (model.path.empty()) {
        set_error(error, "model has no GGUF path");
        status = WEVISION_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto context = std::make_unique<BackendContext>();
    context->kind_ = kind;
    if (!is_supported_backend(kind)) {
        set_error(error, "unknown backend");
        status = WEVISION_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    switch (kind) {
    case WEVISION_BACKEND_CPU:
        context->backend_ = ggml_backend_cpu_init();
        break;
    case WEVISION_BACKEND_METAL:
#if defined(WEVISION_USE_METAL)
        context->backend_ = ggml_backend_metal_init();
        break;
#else
        set_error(error, "Metal support was disabled at build time");
        status = WEVISION_ERROR_INVALID_ARGUMENT;
        return nullptr;
#endif
    case WEVISION_BACKEND_CUDA:
#if defined(WEVISION_USE_CUDA)
        // Device index 0: this standalone runtime targets a single GPU.
        context->backend_ = ggml_backend_cuda_init(0);
        break;
#else
        set_error(error, "CUDA support was disabled at build time");
        status = WEVISION_ERROR_INVALID_ARGUMENT;
        return nullptr;
#endif
    }
    if (context->backend_ == nullptr) {
        set_error(error, "cannot initialize requested ggml backend");
        status = WEVISION_ERROR_INTERNAL;
        return nullptr;
    }

    ggml_context *metadata = nullptr;
    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/&metadata};
    context->gguf_ = gguf_init_from_file(model.path.c_str(), params);
    context->metadata_ = metadata;
    if (context->gguf_ == nullptr || context->metadata_ == nullptr) {
        set_error(error, "cannot reopen GGUF tensor descriptors for backend");
        status = WEVISION_ERROR_MODEL_FORMAT;
        return nullptr;
    }
    const int64_t tensor_count = gguf_get_n_tensors(context->gguf_);
    if (tensor_count != static_cast<int64_t>(model.tensors.size())) {
        set_error(error,
                  "GGUF tensor count changed while creating backend context");
        status = WEVISION_ERROR_MODEL_FORMAT;
        return nullptr;
    }
    for (int64_t index = 0; index < tensor_count; ++index) {
        const char *name = gguf_get_tensor_name(context->gguf_, index);
        ggml_tensor *tensor = name == nullptr
                                  ? nullptr
                                  : ggml_get_tensor(context->metadata_, name);
        if (tensor == nullptr ||
            !context->tensors_.emplace(name, tensor).second) {
            set_error(error, "invalid or duplicate backend tensor descriptor");
            status = WEVISION_ERROR_MODEL_FORMAT;
            return nullptr;
        }
    }

    context->weights_ =
        ggml_backend_alloc_ctx_tensors(context->metadata_, context->backend_);
    if (context->weights_ == nullptr) {
        set_error(error, "cannot allocate backend buffer for model weights");
        status = WEVISION_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }

    std::ifstream input(model.path, std::ios::binary);
    if (!input) {
        set_error(error, "cannot reopen GGUF for tensor upload");
        status = WEVISION_ERROR_FILE_NOT_FOUND;
        return nullptr;
    }
    constexpr size_t kTransferBytes = 16U * 1024U * 1024U;
    std::vector<char> transfer(kTransferBytes);
    const size_t data_offset = gguf_get_data_offset(context->gguf_);
    for (int64_t index = 0; index < tensor_count; ++index) {
        const char *name = gguf_get_tensor_name(context->gguf_, index);
        ggml_tensor *tensor = context->tensors_.at(name);
        const size_t file_offset =
            data_offset + gguf_get_tensor_offset(context->gguf_, index);
        const size_t bytes = ggml_nbytes(tensor);
        input.seekg(static_cast<std::streamoff>(file_offset));
        if (!input) {
            set_error(error, "cannot seek to tensor " + std::string(name));
            status = WEVISION_ERROR_MODEL_FORMAT;
            return nullptr;
        }
        for (size_t written = 0; written < bytes;) {
            const size_t chunk = std::min(kTransferBytes, bytes - written);
            input.read(transfer.data(), static_cast<std::streamsize>(chunk));
            if (input.gcount() != static_cast<std::streamsize>(chunk)) {
                set_error(error, "cannot read tensor " + std::string(name));
                status = WEVISION_ERROR_MODEL_FORMAT;
                return nullptr;
            }
            ggml_backend_tensor_set(tensor, transfer.data(), written, chunk);
            written += chunk;
        }
    }
    status = WEVISION_OK;
    return context;
}

GraphSession::GraphSession(const Model &model, const BackendContext &backend,
                           size_t graph_memory_bytes,
                           ggml_gallocr_t reusable_allocator)
    : model_(model), backend_(backend), graph_memory_bytes_(graph_memory_bytes),
      reusable_allocator_(reusable_allocator) {
    create_context();
}

void GraphSession::create_context() {
    const ggml_init_params params = {
        /*.mem_size   =*/graph_memory_bytes_,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    context_ = ggml_init(params);
}

GraphSession::~GraphSession() {
    if (allocator_ != nullptr && allocator_ != reusable_allocator_)
        ggml_gallocr_free(allocator_);
    if (context_ != nullptr)
        ggml_free(context_);
}

bool GraphSession::ready(std::string *error) const {
    if (context_ == nullptr) {
        *error = "cannot create ggml graph context";
        return false;
    }
    if (!backend_.ready()) {
        *error = "model weights have not been realized on a backend";
        return false;
    }
    return true;
}

bool GraphSession::checkpoint(TraceTensorInfo info, ggml_tensor *tensor,
                              std::string *error) {
    if (trace_ == nullptr || !trace_->wants(info))
        return true;
    // The sink validates the name/tensor pair and rejects duplicates in its
    // watch list; checkpoints_ mirrors that list in the same order, so no
    // second validation is needed here.
    if (!trace_->checkpoint(info, tensor, error))
        return false;
    checkpoints_.push_back({std::move(info), tensor});
    return true;
}

ggml_tensor *GraphSession::weight(const std::string &name,
                                  std::string *error) const {
    return backend_.tensor(name, error);
}

bool GraphSession::compute(ggml_tensor *output, std::string *error,
                           TraceSink *trace) {
    return compute(std::vector<ggml_tensor *>{output}, error, trace);
}

bool GraphSession::compute(const std::vector<ggml_tensor *> &outputs,
                           std::string *error, TraceSink *trace) {
    return compute_impl(outputs, error, trace, nullptr, nullptr);
}

bool GraphSession::compute_profiled(
    ggml_tensor *output, const std::vector<GraphProfilePoint> &points,
    std::vector<GraphProfileSample> *samples, std::string *error) {
    return compute_profiled(std::vector<ggml_tensor *>{output}, points, samples,
                            error);
}

bool GraphSession::compute_profiled(
    const std::vector<ggml_tensor *> &outputs,
    const std::vector<GraphProfilePoint> &points,
    std::vector<GraphProfileSample> *samples, std::string *error) {
    if (samples == nullptr) {
        set_error(error, "profile sample output is null");
        return false;
    }
    samples->clear();
    return compute_impl(outputs, error, nullptr, &points, samples);
}

bool GraphSession::compute_impl(
    const std::vector<ggml_tensor *> &outputs, std::string *error,
    TraceSink *trace, const std::vector<GraphProfilePoint> *profile_points,
    std::vector<GraphProfileSample> *profile_samples) {
    if (!ready(error))
        return false;
    if (outputs.empty() ||
        std::any_of(outputs.begin(), outputs.end(),
                    [](ggml_tensor *output) { return output == nullptr; })) {
        *error = "cannot execute an empty or null graph output set";
        return false;
    }
    if (allocator_ != nullptr) {
        *error = "graph session already contains an executed graph; call reset "
                 "before building another";
        return false;
    }

    TraceSink *active_trace = trace != nullptr ? trace : trace_;
    // Full 28-layer prefill, and especially a diagnostic run that keeps many
    // independent checkpoints alive, exceeds ggml's small default node cap.
    // This is graph metadata only; activation allocation remains backend-led.
    ggml_cgraph *graph =
        ggml_new_graph_custom(context_, GGML_DEFAULT_GRAPH_SIZE * 4U, false);
    // gallocr may reuse an activation as soon as its final graph consumer has
    // run. A synthetic zero-valued dependency makes side branches reachable,
    // but does not keep their storage alive until TraceSink::capture(); in a
    // large alpha graph this allowed a later scalar temporary to overwrite the
    // first allocator block of logits_s1. Build every selected checkpoint as a
    // real graph root and mark it as an output so its buffer remains stable.
    if (active_trace != nullptr) {
        for (const Checkpoint &checkpoint : checkpoints_) {
            ggml_set_output(checkpoint.tensor);
            ggml_build_forward_expand(graph, checkpoint.tensor);
        }
    }
    for (ggml_tensor *output : outputs) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    allocator_ = reusable_allocator_ != nullptr
                     ? reusable_allocator_
                     : ggml_gallocr_new(ggml_backend_get_default_buffer_type(
                           backend_.backend()));
    if (allocator_ == nullptr || !ggml_gallocr_alloc_graph(allocator_, graph)) {
        if (allocator_ != nullptr) {
            if (allocator_ != reusable_allocator_)
                ggml_gallocr_free(allocator_);
            allocator_ = nullptr;
        }
        *error = "cannot allocate graph activations on requested backend";
        return false;
    }
    for (const PendingInput &input : pending_inputs_) {
        if (input.data.size() != ggml_nbytes(input.tensor)) {
            *error = "invalid queued graph input";
            reset();
            return false;
        }
        if (input.tensor->buffer == nullptr) {
            *error =
                "queued graph input is not reachable from the graph output";
            reset();
            return false;
        }
        ggml_backend_tensor_set(input.tensor, input.data.data(), 0,
                                input.data.size());
    }
    int graph_status = GGML_STATUS_SUCCESS;
    if (operation_profile_enabled()) {
        if (!execute_operation_profiled(graph, profile_points, profile_samples,
                                        &graph_status, error))
            return false;
    } else if (profile_points != nullptr) {
        if (!execute_profiled(graph, profile_points, profile_samples,
                              &graph_status, error))
            return false;
    } else {
        graph_status = ggml_backend_graph_compute(backend_.backend(), graph);
    }
    if (graph_status != GGML_STATUS_SUCCESS) {
        *error = "ggml backend failed while computing graph";
        reset();
        return false;
    }
    if (active_trace != nullptr && !active_trace->capture(backend_, error)) {
        return false;
    }
    return true;
}

// Operation-profile execution (WEVISION_OP_PROFILE): pauses the backend at
// every tracked operation to time it in isolation, and interleaves the
// profile-point sampling of execute_profiled() when points are requested.
bool GraphSession::execute_operation_profiled(
    ggml_cgraph *graph, const std::vector<GraphProfilePoint> *profile_points,
    std::vector<GraphProfileSample> *profile_samples, int *graph_status,
    std::string *error) {
    struct OperationSummary {
        size_t count = 0;
        size_t output_bytes = 0;
        double milliseconds = 0.0;
    };

    std::array<OperationSummary, 6> summaries{};
    int begin = 0;
    double profile_interval_ms = 0.0;
    auto compute_range = [&](int range_begin, int range_end,
                             double *milliseconds) {
        if (range_begin >= range_end)
            return GGML_STATUS_SUCCESS;
        ggml_cgraph view = make_graph_view(graph, range_begin, range_end);
        const auto started = std::chrono::steady_clock::now();
        const enum ggml_status range_status =
            ggml_backend_graph_compute(backend_.backend(), &view);
        const auto finished = std::chrono::steady_clock::now();
        if (milliseconds != nullptr) {
            *milliseconds =
                std::chrono::duration<double, std::milli>(finished - started)
                    .count();
        }
        return range_status;
    };
    for (int node_index = 0; node_index < ggml_graph_n_nodes(graph);
         ++node_index) {
        ggml_tensor *node = ggml_graph_node(graph, node_index);
        const int summary_index = operation_profile_index(node->op);
        if (summary_index >= 0) {
            double prelude_ms = 0.0;
            *graph_status = compute_range(begin, node_index, &prelude_ms);
            profile_interval_ms += prelude_ms;
            if (*graph_status != GGML_STATUS_SUCCESS)
                break;

            double operation_ms = 0.0;
            *graph_status =
                compute_range(node_index, node_index + 1, &operation_ms);
            profile_interval_ms += operation_ms;
            if (*graph_status != GGML_STATUS_SUCCESS)
                break;
            begin = node_index + 1;

            OperationSummary &summary = summaries[summary_index];
            ++summary.count;
            summary.output_bytes += ggml_nbytes(node);
            summary.milliseconds += operation_ms;
            std::fprintf(stderr, "op_profile node=%d op=%s", node_index,
                         ggml_op_name(node->op));
            print_profile_tensor("src0", node->src[0]);
            print_profile_tensor("src1", node->src[1]);
            print_profile_tensor("dst", node);
            std::fprintf(stderr, " ms=%.4f\n", operation_ms);
        }

        if (profile_points == nullptr)
            continue;
        const auto found =
            std::find_if(profile_points->begin(), profile_points->end(),
                         [node](const GraphProfilePoint &point) {
                             return point.tensor == node;
                         });
        if (found == profile_points->end())
            continue;
        if (found->name.empty()) {
            *error = "graph profile point has an empty name";
            reset();
            return false;
        }
        double remainder_ms = 0.0;
        *graph_status = compute_range(begin, node_index + 1, &remainder_ms);
        profile_interval_ms += remainder_ms;
        if (*graph_status != GGML_STATUS_SUCCESS)
            break;
        profile_samples->push_back(
            {found->name, profile_interval_ms, found->tag});
        profile_interval_ms = 0.0;
        begin = node_index + 1;
    }
    if (*graph_status == GGML_STATUS_SUCCESS &&
        begin < ggml_graph_n_nodes(graph)) {
        *graph_status =
            compute_range(begin, ggml_graph_n_nodes(graph), nullptr);
    }
    if (*graph_status == GGML_STATUS_SUCCESS && profile_points != nullptr &&
        profile_samples->size() != profile_points->size()) {
        *error = "one or more graph profile points are unreachable";
        reset();
        return false;
    }
    constexpr std::array<const char *, 6> operation_names{
        "PERMUTE", "CONT", "REPEAT", "CONCAT", "IM2COL", "MUL_MAT"};
    for (size_t index = 0; index < summaries.size(); ++index) {
        const OperationSummary &summary = summaries[index];
        std::fprintf(
            stderr,
            "op_profile_summary op=%s count=%zu output_bytes=%zu ms=%.4f\n",
            operation_names[index], summary.count, summary.output_bytes,
            summary.milliseconds);
    }
    return true;
}

// Profile-point execution: computes the graph in node windows between
// consecutive profile points so each requested point receives its own
// interval sample. Failure semantics match execute_operation_profiled().
bool GraphSession::execute_profiled(
    ggml_cgraph *graph, const std::vector<GraphProfilePoint> *profile_points,
    std::vector<GraphProfileSample> *profile_samples, int *graph_status,
    std::string *error) {
    int begin = 0;
    auto previous = std::chrono::steady_clock::now();
    for (int node_index = 0; node_index < ggml_graph_n_nodes(graph);
         ++node_index) {
        ggml_tensor *node = ggml_graph_node(graph, node_index);
        const auto found =
            std::find_if(profile_points->begin(), profile_points->end(),
                         [node](const GraphProfilePoint &point) {
                             return point.tensor == node;
                         });
        if (found == profile_points->end())
            continue;
        if (found->name.empty()) {
            *error = "graph profile point has an empty name";
            reset();
            return false;
        }
        ggml_cgraph view = make_graph_view(graph, begin, node_index + 1);
        *graph_status = ggml_backend_graph_compute(backend_.backend(), &view);
        if (*graph_status != GGML_STATUS_SUCCESS)
            break;
        const auto now = std::chrono::steady_clock::now();
        profile_samples->push_back(
            {found->name,
             std::chrono::duration<double, std::milli>(now - previous).count(),
             found->tag});
        previous = now;
        begin = node_index + 1;
    }
    if (*graph_status == GGML_STATUS_SUCCESS &&
        begin < ggml_graph_n_nodes(graph)) {
        ggml_cgraph tail =
            make_graph_view(graph, begin, ggml_graph_n_nodes(graph));
        *graph_status = ggml_backend_graph_compute(backend_.backend(), &tail);
    }
    if (profile_samples->size() != profile_points->size()) {
        *error = "one or more graph profile points are unreachable";
        reset();
        return false;
    }
    return true;
}

bool GraphSession::set_input(ggml_tensor *tensor, const void *data,
                             size_t bytes, std::string *error) {
    if (tensor == nullptr || data == nullptr || bytes != ggml_nbytes(tensor)) {
        *error = "input payload does not match graph tensor";
        return false;
    }
    if (allocator_ != nullptr) {
        *error = "cannot queue input after graph allocation";
        return false;
    }
    PendingInput input;
    input.tensor = tensor;
    input.data.resize(bytes);
    std::memcpy(input.data.data(), data, bytes);
    pending_inputs_.push_back(std::move(input));
    return true;
}

bool GraphSession::copy_to_host(const ggml_tensor *tensor, void *data,
                                size_t bytes, std::string *error) const {
    if (tensor == nullptr || data == nullptr || bytes != ggml_nbytes(tensor) ||
        tensor->buffer == nullptr) {
        *error = "cannot read unavailable graph tensor";
        return false;
    }
    ggml_backend_tensor_get(tensor, data, 0, bytes);
    return true;
}

bool GraphSession::copy_to_host_range(const ggml_tensor *tensor, void *data,
                                      size_t offset, size_t bytes,
                                      std::string *error) const {
    if (tensor == nullptr || data == nullptr || tensor->buffer == nullptr ||
        offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset) {
        *error = "cannot read graph tensor range";
        return false;
    }
    ggml_backend_tensor_get(tensor, data, offset, bytes);
    return true;
}

void *GraphSession::allocate_custom_data(size_t bytes) {
    if (bytes == 0)
        return nullptr;
    auto storage = std::make_unique<uint8_t[]>(bytes);
    void *result = storage.get();
    custom_data_.push_back(std::move(storage));
    return result;
}

void GraphSession::reset() {
    if (allocator_ != nullptr) {
        if (allocator_ != reusable_allocator_)
            ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
    }
    if (context_ != nullptr) {
        ggml_free(context_);
        context_ = nullptr;
    }
    pending_inputs_.clear();
    checkpoints_.clear();
    custom_data_.clear();
    create_context();
}

} // namespace wevision
