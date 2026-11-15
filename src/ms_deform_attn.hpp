#ifndef WEVISION_MS_DEFORM_ATTN_HPP
#define WEVISION_MS_DEFORM_ATTN_HPP

#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace wevision {

// Backend-neutral ggml graph implementing PyTorch
// grid_sample(..., align_corners=false) for MSDeformAttn. All inputs are FP32
// and contiguous. `value` is [D,H,T],
// locations is [2,P*L,H,Q], weights is [P*L,H,Q], and `anchor` is [D*H,Q].
// The anchor only defines the output shape and graph dependency.
struct MSDeformAttnConfig {
    int32_t head_dim = 0;
    int32_t heads = 0;
    int32_t levels = 0;
    int32_t points = 0;
    std::vector<int32_t> heights;
    std::vector<int32_t> widths;
};

bool build_ms_deform_attn_sampler(GraphSession &session, ggml_tensor *anchor,
                                  ggml_tensor *value, ggml_tensor *locations,
                                  ggml_tensor *weights,
                                  const MSDeformAttnConfig &config,
                                  ggml_tensor **output, std::string *error);

} // namespace wevision

#endif
