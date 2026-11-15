#ifndef WEVISION_PIXEL_OPS_HPP
#define WEVISION_PIXEL_OPS_HPP

#include "runtime.hpp"

#include <cstdint>
#include <string>

namespace wevision {

// Backend-neutral pixel operations used by the full-resolution mask path.
// Tensors use the internal contiguous C,W,H layout. The WHC working layout
// counterparts (resize, pixel shuffle) live in mask.cpp.
bool build_pixel_shuffle2x(GraphSession &session, ggml_tensor *input,
                           ggml_tensor **output, std::string *error);
bool build_bilinear_resize(GraphSession &session, ggml_tensor *input,
                           int32_t target_width, int32_t target_height,
                           ggml_tensor **output, std::string *error);
bool build_adaptive_avg_pool2d(GraphSession &session, ggml_tensor *input,
                               int32_t target_width, int32_t target_height,
                               ggml_tensor **output, std::string *error);

} // namespace wevision
#endif
