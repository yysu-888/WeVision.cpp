#ifndef WEVISION_KV_CACHE_HPP
#define WEVISION_KV_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer *ggml_backend_buffer_t;

namespace wevision {

class BackendContext;

// Request-owned KV storage. Both host and backend-resident modes use ggml's
// contiguous [head_dim, kv_head, position] layout, so appending new positions
// requires no transposition or full-cache upload.
class TextKVCache {
  public:
    TextKVCache() = default;
    ~TextKVCache();

    TextKVCache(const TextKVCache &) = delete;
    TextKVCache &operator=(const TextKVCache &) = delete;

    bool initialize(uint32_t layers, uint32_t capacity, uint32_t kv_heads,
                    uint32_t head_dim, std::string *error);
    bool initialize_backend(const BackendContext &backend, uint32_t layers,
                            uint32_t capacity, uint32_t kv_heads,
                            uint32_t head_dim, std::string *error);
    void reset();

    bool begin_append(uint32_t token_count, std::string *error);
    bool write_layer(uint32_t layer, const float *keys, const float *values,
                     size_t element_count, std::string *error);
    bool mark_layer_written(uint32_t layer, std::string *error);
    bool commit_append(std::string *error);
    // Records an in-place append already performed by the compute graph.
    bool commit_inplace_append(uint32_t token_count, std::string *error);

    uint32_t length() const { return length_; }

    uint32_t capacity() const { return capacity_; }

    uint32_t layer_count() const { return layers_; }

    uint32_t kv_heads() const { return kv_heads_; }

    uint32_t head_dim() const { return head_dim_; }

    size_t layer_elements() const;
    const float *keys(uint32_t layer) const;
    const float *values(uint32_t layer) const;

    bool backend_resident() const { return backend_buffer_ != nullptr; }

    ggml_tensor *key_tensor(uint32_t layer) const;
    ggml_tensor *value_tensor(uint32_t layer) const;

  private:
    bool configure(uint32_t layers, uint32_t capacity, uint32_t kv_heads,
                   uint32_t head_dim, std::string *error);
    void release();

    uint32_t layers_ = 0;
    uint32_t capacity_ = 0;
    uint32_t kv_heads_ = 0;
    uint32_t head_dim_ = 0;
    uint32_t length_ = 0;
    uint32_t pending_tokens_ = 0;
    std::vector<uint8_t> written_;
    std::vector<float> keys_;
    std::vector<float> values_;
    ggml_context *backend_context_ = nullptr;
    ggml_backend_buffer_t backend_buffer_ = nullptr;
    std::vector<ggml_tensor *> key_tensors_;
    std::vector<ggml_tensor *> value_tensors_;
};

} // namespace wevision

#endif
