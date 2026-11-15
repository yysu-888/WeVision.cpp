#include "model_loader.hpp"
#include "runtime.hpp"
#include "text.hpp"
#include "trace.hpp"

#include "ggml.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf TOKEN_IDS.i32 VISION_EMBEDDINGS.f32 "
                     "TRACE_DIR\n",
                     argv[0]);
        return 2;
    }
    wevision_status status = WEVISION_ERROR_INTERNAL;
    auto model = wevision::load_model(argv[1], status);
    std::string error;
    auto backend = model == nullptr
                       ? nullptr
                       : wevision::create_backend_context(
                             *model, WEVISION_BACKEND_CPU, status, &error);
    if (backend == nullptr) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::ifstream ids_file(argv[2], std::ios::binary);
    ids_file.seekg(0, std::ios::end);
    const auto bytes = ids_file.tellg();
    ids_file.seekg(0);
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0)
        return 1;
    std::vector<int32_t> ids(static_cast<size_t>(
        bytes / static_cast<std::streamoff>(sizeof(int32_t))));
    ids_file.read(reinterpret_cast<char *>(ids.data()), bytes);
    std::vector<int32_t> image_positions;
    for (size_t index = 0; index < ids.size(); ++index)
        if (ids[index] == 151655)
            image_positions.push_back(static_cast<int32_t>(index));
    std::vector<float> vision(2048U * image_positions.size());
    std::ifstream vision_file(argv[3], std::ios::binary);
    vision_file.read(
        reinterpret_cast<char *>(vision.data()),
        static_cast<std::streamsize>(vision.size() * sizeof(float)));
    if (!ids_file || !vision_file || image_positions.size() != 12)
        return 1;
    wevision::GraphSession session(*model, *backend, 32U * 1024U * 1024U);
    ggml_tensor *vision_input =
        ggml_new_tensor_2d(session.context(), GGML_TYPE_F32, 2048,
                           static_cast<int64_t>(image_positions.size()));
    if (!session.set_input(vision_input, vision.data(),
                           ggml_nbytes(vision_input), &error))
        return 1;
    wevision::FileTraceSink trace(argv[4], {"text.block.00.*"});
    session.set_trace_sink(&trace);
    ggml_tensor *base = nullptr;
    ggml_tensor *embeddings = nullptr;
    ggml_tensor *norm = nullptr;
    ggml_tensor *q = nullptr;
    ggml_tensor *k = nullptr;
    ggml_tensor *v = nullptr;
    if (!wevision::build_text_input_embeddings(session, ids, image_positions,
                                               vision_input, &base, &embeddings,
                                               &error) ||
        !wevision::build_text_rms_norm(session, embeddings,
                                       "wv.text.blk.0.attn_norm.weight", &norm,
                                       &error) ||
        !wevision::build_text_linear(
            session, norm, "wv.text.blk.0.attn_q.weight", 2048, &q, &error) ||
        !wevision::build_text_linear(
            session, norm, "wv.text.blk.0.attn_k.weight", 1024, &k, &error) ||
        !wevision::build_text_linear(
            session, norm, "wv.text.blk.0.attn_v.weight", 1024, &v, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *q_projected = q;
    ggml_tensor *k_projected = k;
    if (!session.checkpoint({"text.block.00.norm", "text", "B,L,H"}, norm,
                            &error) ||
        !session.checkpoint({"text.block.00.q", "text", "B,L,QH"}, q_projected,
                            &error) ||
        !session.checkpoint({"text.block.00.k", "text", "B,L,KVH"}, k_projected,
                            &error) ||
        !session.checkpoint({"text.block.00.v", "text", "B,L,KVH"}, v,
                            &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const int64_t tokens = q->ne[1];
    q = ggml_reshape_3d(session.context(), q, 128, 16, tokens);
    k = ggml_reshape_3d(session.context(), k, 128, 8, tokens);
    ggml_tensor *q_norm = nullptr;
    ggml_tensor *k_norm = nullptr;
    if (!wevision::build_text_rms_norm(
            session, q, "wv.text.blk.0.attn_q_norm.weight", &q_norm, &error) ||
        !wevision::build_text_rms_norm(
            session, k, "wv.text.blk.0.attn_k_norm.weight", &k_norm, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!session.checkpoint(
            {"text.block.00.q_norm", "text", "B,L,heads,head_dim"}, q_norm,
            &error) ||
        !session.checkpoint(
            {"text.block.00.k_norm", "text", "B,L,kv_heads,head_dim"}, k_norm,
            &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    wevision::TextPositionIds positions;
    wevision::TextRotaryData rotary;
    if (!wevision::build_text_mrope_positions(ids, 151655, 1, 6, 8, 2,
                                              &positions, &error) ||
        !wevision::build_text_rotary_data(positions, 128, 5000000.0F,
                                          {24, 20, 20}, &rotary, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *q_rope = nullptr;
    ggml_tensor *k_rope = nullptr;
    if (!wevision::build_text_qk_rope(session, q_norm, k_norm, rotary, &q_rope,
                                      &k_rope, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!session.checkpoint(
            {"text.block.00.q_rope", "text", "B,L,heads,head_dim"}, q_rope,
            &error) ||
        !session.checkpoint(
            {"text.block.00.k_rope", "text", "B,L,kv_heads,head_dim"}, k_rope,
            &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *value_heads =
        ggml_reshape_3d(session.context(), v, 128, 8, tokens);
    ggml_tensor *attended = nullptr;
    if (!wevision::build_text_causal_attention(
            session, q_rope, k_rope, value_heads, &attended, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    ggml_tensor *attended_flat =
        ggml_reshape_2d(session.context(), attended, 2048, tokens);
    ggml_tensor *attention = nullptr;
    if (!wevision::build_text_linear(session, attended_flat,
                                     "wv.text.blk.0.attn_output.weight", 2048,
                                     &attention, &error) ||
        !session.checkpoint({"text.block.00.attention", "text", "B,L,H"},
                            attention, &error) ||
        !session.compute(attention, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
