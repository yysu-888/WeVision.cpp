#ifndef WEVISION_MASK_OPS_HPP
#define WEVISION_MASK_OPS_HPP

#include "mask.hpp"
#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace wevision {
namespace mask_detail {

// Fourier box encoding in the query builder: 4 box values x 32 frequencies,
// sin and cos each, giving a 256-dim feature per object.
inline constexpr int kBoxFourierFrequencies = 32;
inline constexpr int kBoxFourierDim = 4 * 2 * kBoxFourierFrequencies;

void profile_point(std::vector<GraphProfilePoint> *points, MaskProfileTag tag,
                   std::string name, ggml_tensor *tensor);

ggml_tensor *linear(GraphSession &session, ggml_tensor *x,
                    const std::string &name, int out, std::string *error);

ggml_tensor *linear_parameters(GraphSession &session, ggml_tensor *x,
                               const std::string &weight_name,
                               const std::string &bias_name, int out,
                               std::string *error);

int mask_norm_groups(const GraphSession &session);

ggml_tensor *layer_norm(GraphSession &session, ggml_tensor *x,
                        const std::string &name, std::string *error);

ggml_tensor *conv2d_whc_weight(GraphSession &session, ggml_tensor *x,
                               ggml_tensor *w, int padding, int stride,
                               std::string *error);

ggml_tensor *conv2d_whc(GraphSession &session, ggml_tensor *x,
                        const std::string &weight_name, int padding, int stride,
                        std::string *error);

ggml_tensor *conv2d_whc(GraphSession &session, ggml_tensor *x,
                        const std::string &weight_name, int padding,
                        std::string *error);

ggml_tensor *add_channel_bias_whc(GraphSession &session, ggml_tensor *x,
                                  const std::string &bias_name,
                                  std::string *error);

ggml_tensor *group_norm_affine_whc(GraphSession &session, ggml_tensor *x,
                                   const std::string &name, int groups,
                                   std::string *error);

ggml_tensor *expand_batch(ggml_context *c, ggml_tensor *x, int64_t batch);

ggml_tensor *to_whc(ggml_context *c, ggml_tensor *x);

bool resize_whc(GraphSession &session, ggml_tensor *x, int32_t width,
                int32_t height, ggml_tensor **output, std::string *error);

ggml_tensor *pixel_shuffle2x_whc(GraphSession &session, ggml_tensor *x,
                                 std::string *error);

ggml_tensor *conv_transpose2x_whc(GraphSession &session, ggml_tensor *x,
                                  const std::string &weight_name,
                                  std::string *error);

ggml_tensor *conv_transpose2x_cwh_impl(GraphSession &session, ggml_tensor *x,
                                       const std::string &weight_name,
                                       bool phase_decomposition,
                                       std::string *error);

} // namespace mask_detail
} // namespace wevision

#endif
