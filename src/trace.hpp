#ifndef WEVISION_TRACE_HPP
#define WEVISION_TRACE_HPP

#include "ggml.h"

#include <filesystem>
#include <string>
#include <vector>

namespace wevision {

class BackendContext;

struct TraceTensorInfo {
    std::string name;
    std::string stage;
    std::string layout;
};

// Trace sinks are invoked after a graph has completed. They must not be called
// from a ggml custom-op callback: custom ops may be multi-threaded and their
// input/output tensors may live on a different backend.
class TraceSink {
  public:
    virtual ~TraceSink() = default;
    // Called while the graph is being built. A sink may ignore checkpoints it
    // did not request; it must not execute backend work from this callback.
    virtual bool wants(const TraceTensorInfo &info) const = 0;
    virtual bool checkpoint(TraceTensorInfo info, ggml_tensor *tensor,
                            std::string *error) = 0;
    virtual bool capture(const BackendContext &backend, std::string *error) = 0;
};

// A file sink for C++ graph boundaries. Tensor payloads are raw contiguous
// ggml bytes; manifest.json records shape in row-major (reversed ggml) order,
// dtype, layout, FNV-1a64 hash, and the payload filename. Python reference
// traces use .npy/SHA-256, so a later compare tool normalizes both formats.
class FileTraceSink final : public TraceSink {
  public:
    // Empty selectors means every published checkpoint. A selector ending in
    // `*` matches a semantic-name prefix, e.g. `text.layer.12.*`.
    explicit FileTraceSink(std::filesystem::path root,
                           std::vector<std::string> selectors = {});

    bool wants(const TraceTensorInfo &info) const override;
    bool checkpoint(TraceTensorInfo info, ggml_tensor *tensor,
                    std::string *error) override;
    // Compatibility alias for pre-hook tests. New graph builders use
    // GraphSession::checkpoint() instead.
    bool watch(TraceTensorInfo info, ggml_tensor *tensor, std::string *error);
    bool capture(const BackendContext &backend, std::string *error) override;

  private:
    struct Watch {
        TraceTensorInfo info;
        ggml_tensor *tensor = nullptr;
    };

    std::filesystem::path root_;
    std::vector<std::string> selectors_;
    std::vector<Watch> watches_;
    bool captured_ = false;
};

} // namespace wevision

#endif
