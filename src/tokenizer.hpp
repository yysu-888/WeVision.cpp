#ifndef WEVISION_TOKENIZER_HPP
#define WEVISION_TOKENIZER_HPP

#include "model_loader.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wevision {

// Special tokens resolved through the tokenizer vocabulary at runtime.
inline constexpr const char *kImEndToken = "<|im_end|>";
inline constexpr const char *kImagePadToken = "<|image_pad|>";
inline constexpr const char *kCoordinateBeginToken = "<0>";
inline constexpr const char *kCoordinateEndToken = "<999>";
// The coordinate vocabulary is contiguous <0>..<999>: 1000 discrete levels
// per image axis, shared by generation and response parsing.
inline constexpr int kCoordinateLevels = 1000;

// The runtime representation intentionally owns parsed data.  Model continues
// to own the immutable GGUF strings, so one Model can serve many requests.
struct ChatMessage {
    std::string role;
    std::string content;
    bool has_image = false;
};

class QwenTokenizer {
  public:
    bool load(const TokenizerAssets &assets, std::string *error);
    bool encode(const std::string &text, std::vector<int32_t> *ids,
                std::string *error) const;
    bool decode(const std::vector<int32_t> &ids, std::string *text,
                std::string *error) const;

    // Implements the released Qwen chat template's ordinary (no tools) path.
    // Every image expands to exactly vision_token_count image-pad tokens.
    bool encode_chat(const std::vector<ChatMessage> &messages,
                     bool add_generation_prompt, int32_t vision_token_count,
                     std::vector<int32_t> *ids, std::string *error) const;

    int32_t token_id(const std::string &token) const;

  private:
    bool encode_regular(const std::string &text, std::vector<int32_t> *ids,
                        std::string *error) const;
    bool bpe(const std::string &piece, std::vector<int32_t> *ids,
             std::string *error) const;

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> added_tokens_; // longest-first
    std::string unk_token_;
};

} // namespace wevision

#endif
