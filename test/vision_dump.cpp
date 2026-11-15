#include "vision.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

bool read_token(FILE *file, std::string *token) {
    token->clear();
    int c = 0;
    do {
        c = std::fgetc(file);
        if (c == '#') {
            do {
                c = std::fgetc(file);
            } while (c != '\n' && c != EOF);
        }
    } while (c != EOF && (c == ' ' || c == '\t' || c == '\r' || c == '\n'));
    while (c != EOF && c != ' ' && c != '\t' && c != '\r' && c != '\n') {
        token->push_back(static_cast<char>(c));
        c = std::fgetc(file);
    }
    return !token->empty();
}

bool read_ppm(const char *path, wevision::RgbImage *output) {
    FILE *file = std::fopen(path, "rb");
    if (file == nullptr)
        return false;
    std::string magic, width, height, max_value;
    const bool header = read_token(file, &magic) && read_token(file, &width) &&
                        read_token(file, &height) &&
                        read_token(file, &max_value);
    if (!header || magic != "P6" || max_value != "255") {
        std::fclose(file);
        return false;
    }
    const int image_width = std::atoi(width.c_str());
    const int image_height = std::atoi(height.c_str());
    if (image_width <= 0 || image_height <= 0) {
        std::fclose(file);
        return false;
    }
    wevision::RgbImage image;
    image.width = image_width;
    image.height = image_height;
    image.pixels.resize(static_cast<size_t>(image_width) * image_height * 3U);
    const size_t got =
        std::fread(image.pixels.data(), 1, image.pixels.size(), file);
    std::fclose(file);
    if (got != image.pixels.size())
        return false;
    *output = std::move(image);
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s INPUT.ppm OUTPUT.f32 MIN_PIXELS MAX_PIXELS\n",
                     argv[0]);
        return 2;
    }
    wevision::RgbImage image;
    if (!read_ppm(argv[1], &image)) {
        std::fprintf(stderr, "could not read binary PPM RGB image: %s\n",
                     argv[1]);
        return 1;
    }
    wevision::VisionPreprocessConfig config;
    config.min_pixels = std::atoi(argv[3]);
    config.max_pixels = std::atoi(argv[4]);
    wevision::VisionImageInput prepared;
    if (!wevision::prepare_vision_image(image, config, &prepared)) {
        std::fputs("vision preprocessing failed\n", stderr);
        return 1;
    }
    FILE *file = std::fopen(argv[2], "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "could not open output: %s (%s)\n", argv[2],
                     std::strerror(errno));
        return 1;
    }
    const size_t count = prepared.patches.values.size();
    const bool written = std::fwrite(prepared.patches.values.data(),
                                     sizeof(float), count, file) == count;
    std::fclose(file);
    if (!written) {
        std::fputs("could not write output tensor\n", stderr);
        return 1;
    }
    std::printf("grid_thw=[%d,%d,%d] shape=[%zu,%d]\n", prepared.grid_thw[0],
                prepared.grid_thw[1], prepared.grid_thw[2],
                count / prepared.patches.patch_dim, prepared.patches.patch_dim);
    return 0;
}
