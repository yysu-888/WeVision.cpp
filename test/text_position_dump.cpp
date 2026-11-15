#include "model_loader.hpp"
#include "text.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf OUTPUT.i64\n", argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    if (!model || status != WEVISION_OK) {
        std::fprintf(stderr, "model load failed: %s\n",
                     model ? model->last_error.c_str() : "unknown");
        return 1;
    }
    wevision::QwenTokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(model->tokenizer, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const std::vector<wevision::ChatMessage> messages = {
        {"system", "You are a helpful assistant.", false},
        {"user",
         "Locate the instances that match the following description and "
         "provide "
         "the alpha mattes: person.",
         true},
        {"assistant", "<ref>person</ref><box><10><71><750><1000><OBJ></box>",
         false},
    };
    std::vector<int32_t> ids;
    if (!tokenizer.encode_chat(messages, false, 12, &ids, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    wevision::TextPositionIds positions;
    if (!wevision::build_text_mrope_positions(
            ids, tokenizer.token_id("<|image_pad|>"), 1, 6, 8, 2, &positions,
            &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char *>(positions.values.data()),
                 static_cast<std::streamsize>(positions.values.size() *
                                              sizeof(int64_t)));
    if (!output) {
        std::fputs("failed to write position ids\n", stderr);
        return 1;
    }
    std::printf("wrote [3,%d] M-RoPE position ids\n",
                positions.sequence_length);
    return 0;
}
