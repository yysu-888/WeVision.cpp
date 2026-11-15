#ifndef WEVISION_GRAPH_OPS_HPP
#define WEVISION_GRAPH_OPS_HPP

#include "ggml.h"

#include <stdexcept>
#include <string>

namespace wevision {

class GraphShapeError : public std::runtime_error {
  public:
    explicit GraphShapeError(const std::string &message)
        : std::runtime_error(message) {}
};

inline std::string graph_shape_message(const char *operation,
                                       const ggml_tensor *a,
                                       const ggml_tensor *b,
                                       const char *reason) {
    auto shape = [](const ggml_tensor *tensor) {
        if (!tensor)
            return std::string("null");
        return std::to_string(tensor->ne[0]) + "x" +
               std::to_string(tensor->ne[1]) + "x" +
               std::to_string(tensor->ne[2]) + "x" +
               std::to_string(tensor->ne[3]);
    };
    return std::string("invalid graph shape for ") + operation + ": a=[" +
           shape(a) + "], b=[" + shape(b) + "]: " + reason;
}

inline void validate_binary_operands(const char *operation, ggml_tensor *a,
                                     ggml_tensor *b) {
    if (!a || !b || !ggml_can_repeat(b, a))
        throw GraphShapeError(graph_shape_message(
            operation, a, b, "b is not broadcastable to a"));
}

inline ggml_tensor *checked_add(ggml_context *context, ggml_tensor *a,
                                ggml_tensor *b) {
    validate_binary_operands("add", a, b);
    return ggml_add(context, a, b);
}

inline ggml_tensor *checked_sub(ggml_context *context, ggml_tensor *a,
                                ggml_tensor *b) {
    validate_binary_operands("sub", a, b);
    return ggml_sub(context, a, b);
}

inline ggml_tensor *checked_mul(ggml_context *context, ggml_tensor *a,
                                ggml_tensor *b) {
    validate_binary_operands("mul", a, b);
    return ggml_mul(context, a, b);
}

inline ggml_tensor *checked_div(ggml_context *context, ggml_tensor *a,
                                ggml_tensor *b) {
    validate_binary_operands("div", a, b);
    return ggml_div(context, a, b);
}

inline ggml_tensor *checked_mul_mat(ggml_context *context, ggml_tensor *a,
                                    ggml_tensor *b) {
    if (!a || !b || a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 ||
        b->ne[3] % a->ne[3] != 0 || ggml_is_transposed(a))
        throw GraphShapeError(graph_shape_message(
            "mul_mat", a, b, "inner or batch dimensions are incompatible"));
    return ggml_mul_mat(context, a, b);
}

inline ggml_tensor *checked_repeat(ggml_context *context, ggml_tensor *a,
                                   ggml_tensor *b) {
    if (!a || !b || !ggml_can_repeat(a, b))
        throw GraphShapeError(
            graph_shape_message("repeat", a, b, "a cannot be repeated to b"));
    return ggml_repeat(context, a, b);
}

inline ggml_tensor *checked_concat(ggml_context *context, ggml_tensor *a,
                                   ggml_tensor *b, int dim) {
    bool valid =
        a && b && dim >= 0 && dim < GGML_MAX_DIMS && a->type == b->type;
    for (int axis = 0; valid && axis < GGML_MAX_DIMS; ++axis) {
        if (axis != dim && a->ne[axis] != b->ne[axis])
            valid = false;
    }
    if (!valid)
        throw GraphShapeError(graph_shape_message(
            "concat", a, b, "non-concatenated dimensions or types differ"));
    return ggml_concat(context, a, b, dim);
}

// Conv wrappers mirror the GGML_ASSERT preconditions of their ggml
// counterparts so a mismatched weight produces a GraphShapeError (converted
// to an error return at the inference boundary) instead of a process abort.
// `a` is the kernel and `b` the input, matching ggml naming.
inline ggml_tensor *checked_conv_2d_direct(ggml_context *context,
                                           ggml_tensor *a, ggml_tensor *b,
                                           int s0, int s1, int p0, int p1,
                                           int d0, int d1) {
    if (!a || !b || a->ne[2] != b->ne[2])
        throw GraphShapeError(graph_shape_message(
            "conv_2d_direct", a, b,
            "kernel input channels differ from tensor channels"));
    return ggml_conv_2d_direct(context, a, b, s0, s1, p0, p1, d0, d1);
}

inline ggml_tensor *checked_conv_2d(ggml_context *context, ggml_tensor *a,
                                    ggml_tensor *b, int s0, int s1, int p0,
                                    int p1, int d0, int d1) {
    if (!a || !b || a->ne[2] != b->ne[2])
        throw GraphShapeError(graph_shape_message(
            "conv_2d", a, b,
            "kernel input channels differ from tensor channels"));
    return ggml_conv_2d(context, a, b, s0, s1, p0, p1, d0, d1);
}

inline ggml_tensor *checked_conv_2d_dw_direct(ggml_context *context,
                                              ggml_tensor *a, ggml_tensor *b,
                                              int s0, int s1, int p0, int p1,
                                              int d0, int d1) {
    if (!a || !b || a->ne[2] != 1 || a->ne[3] != b->ne[2])
        throw GraphShapeError(graph_shape_message(
            "conv_2d_dw_direct", a, b,
            "depthwise kernel must be [KW,KH,1,C] matching tensor channels"));
    return ggml_conv_2d_dw_direct(context, a, b, s0, s1, p0, p1, d0, d1);
}

inline ggml_tensor *checked_conv_transpose_2d_p0(ggml_context *context,
                                                 ggml_tensor *a, ggml_tensor *b,
                                                 int stride) {
    if (!a || !b || a->ne[3] != b->ne[2])
        throw GraphShapeError(graph_shape_message(
            "conv_transpose_2d_p0", a, b,
            "kernel output channels differ from tensor channels"));
    return ggml_conv_transpose_2d_p0(context, a, b, stride);
}

} // namespace wevision

#endif
