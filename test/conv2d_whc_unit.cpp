#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct ConvCase {
    int W, H, IC, OC, KH, KW, pad, stride;
};

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::string error;
    auto model = wevision::load_model(argv[1], status);
    auto backend = model ? wevision::create_backend_context(
                               *model, WEVISION_BACKEND_CPU, status, &error)
                         : nullptr;
    if (!backend)
        return 1;
    wevision::GraphSession session(*model, *backend, 128U * 1024U * 1024U);

    // Cases span the mask pipeline: 1x1 fuse convs, 3x3 pad-1 detail convs,
    // and stride-2 downsample convs. All use N == 1 (single object).
    const ConvCase cases[] = {
        {16, 12, 192, 48, 1, 1, 0, 1}, // 1x1 fuse, full channel mix
        {8, 8, 16, 32, 3, 3, 1, 1},    // 3x3 pad-1
        {15, 13, 8, 16, 3, 3, 1, 2},   // 3x3 stride-2
        {6, 5, 4, 4, 1, 1, 0, 1},      // small 1x1
    };

    for (const auto &tc : cases) {
        const int64_t OW = (tc.W + 2 * tc.pad - tc.KW) / tc.stride + 1;
        const int64_t OH = (tc.H + 2 * tc.pad - tc.KH) / tc.stride + 1;
        ggml_context *ctx = session.context();
        // weight: [KH, KW, IC, OC] — ggml conv convention (ne[0]=KH fastest)
        ggml_tensor *weight =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.KH, tc.KW, tc.IC, tc.OC);
        // input: [W, H, IC, 1] — WHC layout, N=1
        ggml_tensor *input =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.W, tc.H, tc.IC, 1);
        std::vector<float> w_data(ggml_nelements(weight));
        std::vector<float> x_data(ggml_nelements(input));
        // Deterministic pseudo-random fill (sin-based, like existing tests).
        for (size_t i = 0; i < w_data.size(); ++i)
            w_data[i] = std::sin(static_cast<float>(i) * 0.013F);
        for (size_t i = 0; i < x_data.size(); ++i)
            x_data[i] = std::sin(static_cast<float>(i) * 0.017F);
        if (!session.set_input(weight, w_data.data(), ggml_nbytes(weight),
                               &error) ||
            !session.set_input(input, x_data.data(), ggml_nbytes(input),
                               &error)) {
            std::fprintf(stderr, "set_input failed: %s\n", error.c_str());
            return 1;
        }
        // Reference: ggml_conv_2d (im2col + mul_mat + reshape + permute + cont)
        ggml_tensor *reference = ggml_conv_2d(ctx, weight, input, tc.stride,
                                              tc.stride, tc.pad, tc.pad, 1, 1);
        // Test: conv2d_whc_weight fast path (im2col + mul_mat + reshape only)
        ggml_tensor *test = nullptr;
        if (!wevision::build_mask_conv2d_whc_fixture(
                session, input, weight, tc.pad, tc.stride, &test, &error)) {
            std::fprintf(stderr, "fixture failed: %s\n", error.c_str());
            return 1;
        }
        if (!ggml_are_same_shape(reference, test)) {
            std::fprintf(
                stderr,
                "shape mismatch: ref=[%lld,%lld,%lld,%lld] "
                "test=[%lld,%lld,%lld,%lld]\n",
                (long long)reference->ne[0], (long long)reference->ne[1],
                (long long)reference->ne[2], (long long)reference->ne[3],
                (long long)test->ne[0], (long long)test->ne[1],
                (long long)test->ne[2], (long long)test->ne[3]);
            return 1;
        }
        // Compute abs(reference - test) in the same graph.
        ggml_tensor *diff = ggml_abs(ctx, ggml_sub(ctx, reference, test));
        if (!session.compute({diff}, &error)) {
            std::fprintf(stderr, "compute failed: %s\n", error.c_str());
            return 1;
        }
        std::vector<float> diff_data(ggml_nelements(diff));
        if (!session.copy_to_host(diff, diff_data.data(), ggml_nbytes(diff),
                                  &error)) {
            std::fprintf(stderr, "copy_to_host failed: %s\n", error.c_str());
            return 1;
        }
        float maximum = 0.0F;
        double sum = 0.0;
        for (size_t i = 0; i < diff_data.size(); ++i) {
            maximum = std::max(maximum, diff_data[i]);
            sum += diff_data[i];
        }
        std::printf("case W=%d H=%d IC=%d OC=%d K=%dx%d pad=%d stride=%d "
                    "OW=%lld OH=%lld: max_abs=%.9g mean_abs=%.9g\n",
                    tc.W, tc.H, tc.IC, tc.OC, tc.KH, tc.KW, tc.pad, tc.stride,
                    (long long)OW, (long long)OH, maximum,
                    sum / diff_data.size());
        if (maximum > 1e-5F) {
            std::fprintf(stderr, "FAIL: max_abs exceeds tolerance\n");
            return 1;
        }
        session.reset();
    }
    std::printf("all cases passed\n");
    return 0;
}
