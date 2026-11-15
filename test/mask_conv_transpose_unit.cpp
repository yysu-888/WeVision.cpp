#include "mask.hpp"
#include "model_loader.hpp"
#include "runtime.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::string error;
    auto model = wevision::load_model(argv[1], status);
    auto backend = model ? wevision::create_backend_context(
                               *model, WEVISION_BACKEND_METAL, status, &error)
                         : nullptr;
    if (!backend)
        return 1;
    wevision::GraphSession session(*model, *backend, 64U * 1024U * 1024U);
    constexpr int channels = 48, width = 5, height = 4;
    std::vector<float> values(channels * width * height);
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = std::sin(static_cast<float>(index) * 0.017F);
    ggml_tensor *input = ggml_new_tensor_3d(session.context(), GGML_TYPE_F32,
                                            channels, width, height);
    if (!session.set_input(input, values.data(), ggml_nbytes(input), &error))
        return 1;
    ggml_tensor *reference = nullptr;
    ggml_tensor *phase = nullptr;
    if (!wevision::build_mask_conv_transpose2x_fixture(
            session, input, "wv.mask.alpha.up_s2_to_s1.weight", false,
            &reference, &error) ||
        !wevision::build_mask_conv_transpose2x_fixture(
            session, input, "wv.mask.alpha.up_s2_to_s1.weight", true, &phase,
            &error) ||
        !session.compute({reference, phase}, &error))
        return 1;
    std::vector<float> expected(ggml_nelements(reference));
    std::vector<float> actual(expected.size());
    if (!session.copy_to_host(reference, expected.data(),
                              ggml_nbytes(reference), &error) ||
        !session.copy_to_host(phase, actual.data(), ggml_nbytes(phase), &error))
        return 1;
    float maximum = 0.0F;
    double mean = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::abs(expected[index] - actual[index]);
        maximum = std::max(maximum, difference);
        mean += difference;
    }
    mean /= actual.size();
    std::printf("max_abs=%.9g mean_abs=%.9g\n", maximum, mean);
    return maximum <= 2e-5F ? 0 : 1;
}
