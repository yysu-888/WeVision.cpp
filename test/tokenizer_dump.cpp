#include "model_loader.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf OUTPUT.i32\n", argv[0]);
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
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
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
        std::fprintf(stderr, "tokenizer encode failed: %s\n", error.c_str());
        return 1;
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char *>(ids.data()),
                 static_cast<std::streamsize>(ids.size() * sizeof(int32_t)));
    if (!output) {
        std::fputs("failed to write token ids\n", stderr);
        return 1;
    }
    std::printf("wrote %zu token ids; <OBJ>=%d\n", ids.size(),
                tokenizer.token_id("<OBJ>"));
    return 0;
}
