#include "kv_cache.hpp"
#include "runtime.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <limits>

namespace wevision {

size_t TextKVCache::layer_elements() const {
    return static_cast<size_t>(capacity_) * kv_heads_ * head_dim_;
}

TextKVCache::~TextKVCache() { release(); }

void TextKVCache::release() {
    if (backend_buffer_ != nullptr) {
        ggml_backend_buffer_free(backend_buffer_);
        backend_buffer_ = nullptr;
    }
    if (backend_context_ != nullptr) {
        ggml_free(backend_context_);
        backend_context_ = nullptr;
    }
    key_tensors_.clear();
    value_tensors_.clear();
    keys_.clear();
    values_.clear();
}

bool TextKVCache::configure(uint32_t layers, uint32_t capacity,
                            uint32_t kv_heads, uint32_t head_dim,
                            std::string *error) {
    release();
    if (layers == 0 || capacity == 0 || kv_heads == 0 || head_dim == 0 ||
        static_cast<size_t>(capacity) >
            std::numeric_limits<size_t>::max() / kv_heads / head_dim ||
        static_cast<size_t>(layers) >
            std::numeric_limits<size_t>::max() /
                (static_cast<size_t>(capacity) * kv_heads * head_dim)) {
        set_error(error, "invalid KV cache dimensions");
        return false;
    }
    layers_ = layers;
    capacity_ = capacity;
    kv_heads_ = kv_heads;
    head_dim_ = head_dim;
    length_ = 0;
    pending_tokens_ = 0;
    written_.assign(layers, 0);
    return true;
}

bool TextKVCache::initialize(uint32_t layers, uint32_t capacity,
                             uint32_t kv_heads, uint32_t head_dim,
                             std::string *error) {
    if (!configure(layers, capacity, kv_heads, head_dim, error))
        return false;
    keys_.assign(static_cast<size_t>(layers) * layer_elements(), 0.0F);
    values_.assign(keys_.size(), 0.0F);
    return true;
}

bool TextKVCache::initialize_backend(const BackendContext &backend,
                                     uint32_t layers, uint32_t capacity,
                                     uint32_t kv_heads, uint32_t head_dim,
                                     std::string *error) {
    if (!backend.ready()) {
        set_error(error, "KV cache backend is not initialized");
        return false;
    }
    if (!configure(layers, capacity, kv_heads, head_dim, error))
        return false;
    const size_t metadata_bytes =
        std::max<size_t>(1024U * 1024U, static_cast<size_t>(layers) * 4U *
                                            ggml_tensor_overhead());
    backend_context_ = ggml_init({metadata_bytes, nullptr, true});
    if (backend_context_ == nullptr) {
        set_error(error, "cannot create backend KV cache context");
        release();
        return false;
    }
    key_tensors_.reserve(layers);
    value_tensors_.reserve(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        key_tensors_.push_back(ggml_new_tensor_3d(
            backend_context_, GGML_TYPE_F16, head_dim, kv_heads, capacity));
        value_tensors_.push_back(ggml_new_tensor_3d(
            backend_context_, GGML_TYPE_F16, head_dim, kv_heads, capacity));
    }
    backend_buffer_ =
        ggml_backend_alloc_ctx_tensors(backend_context_, backend.backend());
    if (backend_buffer_ == nullptr) {
        set_error(error, "cannot allocate backend KV cache");
        release();
        return false;
    }
    return true;
}

void TextKVCache::reset() {
    length_ = 0;
    pending_tokens_ = 0;
    std::fill(written_.begin(), written_.end(), 0);
}

bool TextKVCache::begin_append(uint32_t token_count, std::string *error) {
    if (layers_ == 0 || token_count == 0 || pending_tokens_ != 0 ||
        token_count > capacity_ - length_) {
        set_error(error, "invalid KV cache append range");
        return false;
    }
    pending_tokens_ = token_count;
    std::fill(written_.begin(), written_.end(), 0);
    return true;
}

bool TextKVCache::write_layer(uint32_t layer, const float *keys,
                              const float *values, size_t element_count,
                              std::string *error) {
    const size_t expected =
        static_cast<size_t>(pending_tokens_) * kv_heads_ * head_dim_;
    if (pending_tokens_ == 0 || layer >= layers_ || keys == nullptr ||
        values == nullptr || element_count != expected ||
        written_[layer] != 0) {
        set_error(error, "invalid KV cache layer write");
        return false;
    }
    const size_t token_offset =
        static_cast<size_t>(length_) * kv_heads_ * head_dim_;
    if (backend_resident()) {
        ggml_backend_tensor_set(key_tensors_[layer], keys,
                                token_offset * sizeof(float),
                                expected * sizeof(float));
        ggml_backend_tensor_set(value_tensors_[layer], values,
                                token_offset * sizeof(float),
                                expected * sizeof(float));
    } else {
        const size_t base =
            static_cast<size_t>(layer) * layer_elements() + token_offset;
        std::copy_n(keys, expected, keys_.data() + base);
        std::copy_n(values, expected, values_.data() + base);
    }
    written_[layer] = 1;
    return true;
}

bool TextKVCache::mark_layer_written(uint32_t layer, std::string *error) {
    if (!backend_resident() || pending_tokens_ == 0 || layer >= layers_ ||
        written_[layer] != 0) {
        set_error(error, "invalid backend KV cache layer commit");
        return false;
    }
    written_[layer] = 1;
    return true;
}

bool TextKVCache::commit_append(std::string *error) {
    if (pending_tokens_ == 0 ||
        std::find(written_.begin(), written_.end(), 0) != written_.end()) {
        set_error(error, "cannot commit incomplete KV cache append");
        return false;
    }
    length_ += pending_tokens_;
    pending_tokens_ = 0;
    return true;
}

bool TextKVCache::commit_inplace_append(uint32_t token_count,
                                        std::string *error) {
    if (!backend_resident() || !begin_append(token_count, error))
        return false;
    std::fill(written_.begin(), written_.end(), 1);
    return commit_append(error);
}

const float *TextKVCache::keys(uint32_t layer) const {
    return !backend_resident() && layer < layers_
               ? keys_.data() + static_cast<size_t>(layer) * layer_elements()
               : nullptr;
}

const float *TextKVCache::values(uint32_t layer) const {
    return !backend_resident() && layer < layers_
               ? values_.data() + static_cast<size_t>(layer) * layer_elements()
               : nullptr;
}

ggml_tensor *TextKVCache::key_tensor(uint32_t layer) const {
    return backend_resident() && layer < key_tensors_.size()
               ? key_tensors_[layer]
               : nullptr;
}

ggml_tensor *TextKVCache::value_tensor(uint32_t layer) const {
    return backend_resident() && layer < value_tensors_.size()
               ? value_tensors_[layer]
               : nullptr;
}

} // namespace wevision
