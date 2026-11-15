#include "model_loader.hpp"
#include "runtime.hpp"
#include "trace.hpp"
#include "vision.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf TRACE_DIR [PATCHES.f32]\n",
                     argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    std::unique_ptr<wevision::Model> model =
        wevision::load_model(argv[1], status);
    std::string error;
    std::unique_ptr<wevision::BackendContext> backend =
        model == nullptr ? nullptr
                         : wevision::create_backend_context(
                               *model, WEVISION_BACKEND_CPU, status, &error);
    if (backend == nullptr) {
        std::fprintf(stderr, "Cannot create CPU backend: %s\n", error.c_str());
        return 1;
    }
    wevision::PatchifiedImage input;
    input.grid_h = 6;
    input.grid_w = 8;
    input.patch_dim = 1536;
    input.values.assign(48U * 1536U, 0.0F);
    if (argc == 4) {
        std::ifstream payload(argv[3], std::ios::binary | std::ios::ate);
        if (!payload || static_cast<size_t>(payload.tellg()) !=
                            input.values.size() * sizeof(float)) {
            std::fprintf(stderr,
                         "Patch payload must be exactly %zu FP32 values: %s\n",
                         input.values.size(), argv[3]);
            return 1;
        }
        payload.seekg(0);
        payload.read(
            reinterpret_cast<char *>(input.values.data()),
            static_cast<std::streamsize>(input.values.size() * sizeof(float)));
        if (!payload) {
            std::fprintf(stderr, "Cannot read patch payload: %s\n", argv[3]);
            return 1;
        }
    }
    wevision::GraphSession session(*model, *backend, 32U * 1024U * 1024U);
    ggml_tensor *output = nullptr;
    if (!wevision::build_vision_patch_embed(session, input, &output, &error)) {
        std::fprintf(stderr, "Cannot build patch embedding: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *patch_output = output;
    wevision::VisionPositionData positions;
    if (!wevision::build_vision_position_data(input.grid_h, input.grid_w, 2, 48,
                                              &positions) ||
        !wevision::build_vision_positioned_input(session, output, positions,
                                                 &output, &error)) {
        std::fprintf(stderr, "Cannot build positioned patch embedding: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *positioned_output = output;
    ggml_tensor *norm1_output = nullptr;
    if (!wevision::build_vision_layer_norm(
            session, positioned_output, "wv.vision.blk.0.attn_norm.weight",
            "wv.vision.blk.0.attn_norm.bias", &norm1_output, &error)) {
        std::fprintf(stderr, "Cannot build vision block 0 norm1: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *qkv_output = nullptr;
    if (!wevision::build_vision_linear(
            session, norm1_output, "wv.vision.blk.0.attn_qkv.weight",
            "wv.vision.blk.0.attn_qkv.bias", 3072, &qkv_output, &error)) {
        std::fprintf(stderr, "Cannot build vision block 0 qkv: %s\n",
                     error.c_str());
        return 1;
    }
    wevision::VisionRotaryData rotary;
    if (!wevision::build_vision_rotary_data(positions, 64, 10000.0F, &rotary)) {
        std::fputs("Cannot build vision rotary data\n", stderr);
        return 1;
    }
    wevision::VisionAttentionQKV qkv;
    if (!wevision::build_vision_qkv_rotary(session, qkv_output, rotary, &qkv,
                                           &error)) {
        std::fprintf(stderr, "Cannot build vision QKV rotary: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *attention_output = nullptr;
    if (!wevision::build_vision_attention(session, qkv, &attention_output,
                                          &error)) {
        std::fprintf(stderr, "Cannot build vision attention: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *attention_flat = ggml_reshape_2d(
        session.context(), attention_output, 1024, input.grid_h * input.grid_w);
    ggml_tensor *attention_projected = nullptr;
    if (!wevision::build_vision_linear(session, attention_flat,
                                       "wv.vision.blk.0.attn_output.weight",
                                       "wv.vision.blk.0.attn_output.bias", 1024,
                                       &attention_projected, &error)) {
        std::fprintf(stderr, "Cannot build vision attention projection: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *after_attention =
        ggml_add(session.context(), positioned_output, attention_projected);
    ggml_tensor *norm2_output = nullptr;
    if (!wevision::build_vision_layer_norm(
            session, after_attention, "wv.vision.blk.0.ffn_norm.weight",
            "wv.vision.blk.0.ffn_norm.bias", &norm2_output, &error)) {
        std::fprintf(stderr, "Cannot build vision block 0 norm2: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *ffn_up = nullptr;
    if (!wevision::build_vision_linear(
            session, norm2_output, "wv.vision.blk.0.ffn_up.weight",
            "wv.vision.blk.0.ffn_up.bias", 4096, &ffn_up, &error)) {
        std::fprintf(stderr, "Cannot build vision block 0 ffn up: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *ffn_down = nullptr;
    if (!wevision::build_vision_linear(
            session, ggml_gelu(session.context(), ffn_up),
            "wv.vision.blk.0.ffn_down.weight", "wv.vision.blk.0.ffn_down.bias",
            1024, &ffn_down, &error)) {
        std::fprintf(stderr, "Cannot build vision block 0 ffn down: %s\n",
                     error.c_str());
        return 1;
    }
    ggml_tensor *block0_output =
        ggml_add(session.context(), after_attention, ffn_down);
    wevision::FileTraceSink trace(argv[2]);
    if (!trace.watch({"vision.patch_embed.output", "vision", "patch,hidden"},
                     patch_output, &error) ||
        !trace.watch({"vision.positioned_input", "vision", "patch,hidden"},
                     output, &error) ||
        !trace.watch({"vision.block.00.norm1", "vision", "patch,hidden"},
                     norm1_output, &error) ||
        !trace.watch({"vision.block.00.qkv", "vision", "patch,QKV_hidden"},
                     qkv_output, &error) ||
        !trace.watch(
            {"vision.block.00.q_rope", "vision", "patch,heads,head_dim"},
            qkv.query, &error) ||
        !trace.watch(
            {"vision.block.00.k_rope", "vision", "patch,heads,head_dim"},
            qkv.key, &error) ||
        !trace.watch({"vision.block.00.attention_pre_proj", "vision",
                      "patch,heads,head_dim"},
                     attention_output, &error) ||
        !trace.watch({"vision.block.00.attn_output", "vision", "patch,hidden"},
                     attention_projected, &error) ||
        !trace.watch({"vision.block.00.output", "vision", "patch,hidden"},
                     block0_output, &error) ||
        !session.compute(block0_output, &error, &trace)) {
        std::fprintf(stderr, "Cannot compute patch embedding: %s\n",
                     error.c_str());
        return 1;
    }
    std::puts("vision patch embedding graph completed");
    return 0;
}
