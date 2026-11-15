#include "tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace wevision {
namespace {

bool parse_string(const std::string &source, size_t *cursor,
                  std::string *value) {
    if (cursor == nullptr || value == nullptr || *cursor >= source.size() ||
        source[*cursor] != '"')
        return false;
    ++*cursor;
    value->clear();
    while (*cursor < source.size()) {
        const char ch = source[(*cursor)++];
        if (ch == '"')
            return true;
        if (ch != '\\') {
            value->push_back(ch);
            continue;
        }
        if (*cursor >= source.size())
            return false;
        const char escape = source[(*cursor)++];
        switch (escape) {
        case '"':
            value->push_back('"');
            break;
        case '\\':
            value->push_back('\\');
            break;
        case '/':
            value->push_back('/');
            break;
        case 'b':
            value->push_back('\b');
            break;
        case 'f':
            value->push_back('\f');
            break;
        case 'n':
            value->push_back('\n');
            break;
        case 'r':
            value->push_back('\r');
            break;
        case 't':
            value->push_back('\t');
            break;
        case 'u': {
            if (*cursor + 4 > source.size())
                return false;
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
                const char digit = source[*cursor + static_cast<size_t>(i)];
                code <<= 4;
                if (digit >= '0' && digit <= '9')
                    code |= static_cast<unsigned>(digit - '0');
                else if (digit >= 'a' && digit <= 'f')
                    code |= static_cast<unsigned>(digit - 'a' + 10);
                else if (digit >= 'A' && digit <= 'F')
                    code |= static_cast<unsigned>(digit - 'A' + 10);
                else
                    return false;
            }
            *cursor += 4;
            // JSON surrogate pairs are needed for the small number of emoji
            // tokens.
            if (code >= 0xd800U && code <= 0xdbffU &&
                *cursor + 6 <= source.size() &&
                source.compare(*cursor, 2, "\\u") == 0) {
                size_t low_cursor = *cursor + 2;
                unsigned low = 0;
                for (int i = 0; i < 4; ++i) {
                    const char digit =
                        source[low_cursor + static_cast<size_t>(i)];
                    low <<= 4;
                    if (digit >= '0' && digit <= '9')
                        low |= static_cast<unsigned>(digit - '0');
                    else if (digit >= 'a' && digit <= 'f')
                        low |= static_cast<unsigned>(digit - 'a' + 10);
                    else if (digit >= 'A' && digit <= 'F')
                        low |= static_cast<unsigned>(digit - 'A' + 10);
                    else
                        return false;
                }
                if (low >= 0xdc00U && low <= 0xdfffU) {
                    code =
                        0x10000U + ((code - 0xd800U) << 10U) + (low - 0xdc00U);
                    *cursor += 6;
                }
            }
            if (code <= 0x7fU)
                value->push_back(static_cast<char>(code));
            else if (code <= 0x7ffU) {
                value->push_back(static_cast<char>(0xc0U | (code >> 6U)));
                value->push_back(static_cast<char>(0x80U | (code & 0x3fU)));
            } else if (code <= 0xffffU) {
                value->push_back(static_cast<char>(0xe0U | (code >> 12U)));
                value->push_back(
                    static_cast<char>(0x80U | ((code >> 6U) & 0x3fU)));
                value->push_back(static_cast<char>(0x80U | (code & 0x3fU)));
            } else {
                value->push_back(static_cast<char>(0xf0U | (code >> 18U)));
                value->push_back(
                    static_cast<char>(0x80U | ((code >> 12U) & 0x3fU)));
                value->push_back(
                    static_cast<char>(0x80U | ((code >> 6U) & 0x3fU)));
                value->push_back(static_cast<char>(0x80U | (code & 0x3fU)));
            }
            break;
        }
        default:
            return false;
        }
    }
    return false;
}

void skip_space(const std::string &source, size_t *cursor) {
    while (*cursor < source.size() &&
           std::isspace(static_cast<unsigned char>(source[*cursor])))
        ++*cursor;
}

bool parse_number(const std::string &source, size_t *cursor, int32_t *value) {
    skip_space(source, cursor);
    char *end = nullptr;
    const long result = std::strtol(source.c_str() + *cursor, &end, 10);
    if (end == source.c_str() + *cursor || result < INT32_MIN ||
        result > INT32_MAX)
        return false;
    *cursor = static_cast<size_t>(end - source.c_str());
    *value = static_cast<int32_t>(result);
    return true;
}

size_t matching(const std::string &source, size_t open, char left, char right) {
    if (open >= source.size() || source[open] != left)
        return std::string::npos;
    int depth = 0;
    bool in_string = false;
    for (size_t i = open; i < source.size(); ++i) {
        if (in_string) {
            if (source[i] == '\\')
                ++i;
            else if (source[i] == '"')
                in_string = false;
            continue;
        }
        if (source[i] == '"')
            in_string = true;
        else if (source[i] == left)
            ++depth;
        else if (source[i] == right && --depth == 0)
            return i;
    }
    return std::string::npos;
}

bool utf8_at(const std::string &input, size_t offset, uint32_t *codepoint,
             size_t *width) {
    if (offset >= input.size())
        return false;
    const auto first = static_cast<unsigned char>(input[offset]);
    if (first < 0x80U) {
        *codepoint = first;
        *width = 1;
        return true;
    }
    int count = first < 0xe0U ? 2 : first < 0xf0U ? 3 : first < 0xf8U ? 4 : 0;
    if (count == 0 || offset + static_cast<size_t>(count) > input.size())
        return false;
    uint32_t value = first & ((1U << (7 - count)) - 1U);
    for (int i = 1; i < count; ++i) {
        const auto next =
            static_cast<unsigned char>(input[offset + static_cast<size_t>(i)]);
        if ((next & 0xc0U) != 0x80U)
            return false;
        value = (value << 6U) | (next & 0x3fU);
    }
    *codepoint = value;
    *width = static_cast<size_t>(count);
    return true;
}

void append_utf8(std::string *output, uint32_t codepoint) {
    if (codepoint <= 0x7fU)
        output->push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ffU) {
        output->push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output->push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output->push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output->push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        output->push_back(
            static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        output->push_back(
            static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

bool is_space(uint32_t codepoint) {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\r' ||
           codepoint == '\n';
}

bool is_newline(uint32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

bool is_digit(uint32_t codepoint) {
    return codepoint >= '0' && codepoint <= '9';
}

bool is_letter(uint32_t codepoint) {
    return (codepoint >= 'A' && codepoint <= 'Z') ||
           (codepoint >= 'a' && codepoint <= 'z') ||
           codepoint >= 0x80U; // Qwen's regex treats all common non-ASCII
                               // script characters as letters.
}

std::array<uint32_t, 256> byte_to_codepoint() {
    std::array<uint32_t, 256> result{};
    std::array<bool, 256> used{};
    for (int value = 33; value <= 126; ++value) {
        result[static_cast<size_t>(value)] = static_cast<uint32_t>(value);
        used[static_cast<size_t>(value)] = true;
    }
    for (int value = 161; value <= 172; ++value) {
        result[static_cast<size_t>(value)] = static_cast<uint32_t>(value);
        used[static_cast<size_t>(value)] = true;
    }
    for (int value = 174; value <= 255; ++value) {
        result[static_cast<size_t>(value)] = static_cast<uint32_t>(value);
        used[static_cast<size_t>(value)] = true;
    }
    uint32_t next = 256;
    for (int value = 0; value < 256; ++value)
        if (!used[static_cast<size_t>(value)])
            result[static_cast<size_t>(value)] = next++;
    return result;
}

const std::array<uint32_t, 256> k_byte_to_codepoint = byte_to_codepoint();

std::string byte_encode(const std::string &input) {
    std::string result;
    result.reserve(input.size() * 2U);
    for (unsigned char byte : input)
        append_utf8(&result, k_byte_to_codepoint[byte]);
    return result;
}

std::string byte_decode(const std::string &input) {
    std::array<int16_t, 512> reverse{};
    reverse.fill(-1);
    for (size_t byte = 0; byte < k_byte_to_codepoint.size(); ++byte)
        reverse[k_byte_to_codepoint[byte]] = static_cast<int16_t>(byte);
    std::string result;
    for (size_t offset = 0; offset < input.size();) {
        uint32_t cp = 0;
        size_t width = 0;
        if (!utf8_at(input, offset, &cp, &width) || cp >= reverse.size() ||
            reverse[cp] < 0)
            return input;
        result.push_back(static_cast<char>(reverse[cp]));
        offset += width;
    }
    return result;
}

} // namespace

bool QwenTokenizer::load(const TokenizerAssets &assets, std::string *error) {
    token_to_id_.clear();
    id_to_token_.clear();
    merge_rank_.clear();
    added_tokens_.clear();
    const std::string &json = assets.tokenizer_json;
    const size_t model_key = json.find("\"model\"");
    const size_t vocab_key = json.find("\"vocab\"", model_key);
    const size_t merges_key = json.find("\"merges\"", vocab_key);
    if (model_key == std::string::npos || vocab_key == std::string::npos) {
        *error = "tokenizer.json has no model vocab";
        return false;
    }
    const size_t vocab_open = json.find('{', vocab_key);
    // HF's BPE schema places `merges` immediately after `vocab`.  Finding its
    // preceding close brace avoids relying on a general JSON object walker for
    // the 150k-entry, escape-heavy vocabulary map.
    const size_t vocab_end = merges_key == std::string::npos
                                 ? std::string::npos
                                 : json.rfind('}', merges_key);
    if (vocab_open == std::string::npos || vocab_end == std::string::npos) {
        *error = "malformed tokenizer vocabulary";
        return false;
    }
    size_t cursor = vocab_open + 1;
    while (cursor < vocab_end) {
        skip_space(json, &cursor);
        if (cursor >= vocab_end)
            break;
        if (json[cursor] == ',') {
            ++cursor;
            continue;
        }
        std::string token;
        if (!parse_string(json, &cursor, &token)) {
            *error = "malformed tokenizer vocabulary token";
            return false;
        }
        skip_space(json, &cursor);
        if (cursor >= vocab_end || json[cursor++] != ':') {
            *error = "malformed tokenizer vocabulary separator";
            return false;
        }
        int32_t id = 0;
        if (!parse_number(json, &cursor, &id)) {
            *error = "malformed tokenizer vocabulary id";
            return false;
        }
        token_to_id_[token] = id;
        id_to_token_[id] = token;
    }
    const size_t merges_open = json.find('[', merges_key);
    const size_t merges_end = matching(json, merges_open, '[', ']');
    if (merges_key == std::string::npos || merges_open == std::string::npos ||
        merges_end == std::string::npos) {
        *error = "tokenizer.json has no BPE merges";
        return false;
    }
    cursor = merges_open + 1;
    int32_t rank = 0;
    while (cursor < merges_end) {
        skip_space(json, &cursor);
        if (cursor >= merges_end)
            break;
        if (json[cursor] == ',') {
            ++cursor;
            continue;
        }
        std::string left, right;
        if (cursor < merges_end && json[cursor] == '[') {
            ++cursor;
            skip_space(json, &cursor);
            if (!parse_string(json, &cursor, &left)) {
                *error = "malformed BPE merge left token";
                return false;
            }
            skip_space(json, &cursor);
            if (cursor >= merges_end || json[cursor++] != ',') {
                *error = "malformed BPE merge delimiter";
                return false;
            }
            skip_space(json, &cursor);
            if (!parse_string(json, &cursor, &right)) {
                *error = "malformed BPE merge right token";
                return false;
            }
            skip_space(json, &cursor);
            if (cursor >= merges_end || json[cursor++] != ']') {
                *error = "malformed BPE merge array";
                return false;
            }
        } else {
            std::string merge;
            if (!parse_string(json, &cursor, &merge)) {
                *error = "malformed BPE merge";
                return false;
            }
            const size_t split = merge.find(' ');
            if (split == std::string::npos) {
                *error = "BPE merge has no pair separator";
                return false;
            }
            left = merge.substr(0, split);
            right = merge.substr(split + 1);
        }
        merge_rank_[left + '\x1f' + right] = rank++;
    }
    const size_t added_key = json.find("\"added_tokens\"");
    const size_t added_open = json.find('[', added_key);
    const size_t added_end = matching(json, added_open, '[', ']');
    if (added_key == std::string::npos || added_open == std::string::npos ||
        added_end == std::string::npos) {
        *error = "tokenizer.json has no added tokens";
        return false;
    }
    cursor = added_open + 1;
    while (cursor < added_end) {
        skip_space(json, &cursor);
        if (cursor >= added_end)
            break;
        if (json[cursor] == ',') {
            ++cursor;
            continue;
        }
        if (cursor >= added_end || json[cursor] != '{') {
            *error = "malformed added token";
            return false;
        }
        const size_t end = matching(json, cursor, '{', '}');
        if (end == std::string::npos) {
            *error = "unterminated added token";
            return false;
        }
        const std::string object = json.substr(cursor, end - cursor + 1);
        cursor = end + 1;
        const size_t id_key = object.find("\"id\"");
        const size_t content_key = object.find("\"content\"");
        if (id_key == std::string::npos || content_key == std::string::npos) {
            *error = "added token missing id/content";
            return false;
        }
        size_t id_cursor = object.find(':', id_key) + 1;
        int32_t id = 0;
        size_t content_cursor = object.find(':', content_key) + 1;
        skip_space(object, &content_cursor);
        std::string content;
        if (!parse_number(object, &id_cursor, &id) ||
            !parse_string(object, &content_cursor, &content)) {
            *error = "invalid added token";
            return false;
        }
        token_to_id_[content] = id;
        id_to_token_[id] = content;
        added_tokens_.emplace_back(content, id);
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const auto &a, const auto &b) {
                  return a.first.size() > b.first.size();
              });
    const size_t unk_key = json.find("\"unk_token\"");
    if (unk_key != std::string::npos) {
        size_t value = json.find(':', unk_key) + 1;
        skip_space(json, &value);
        parse_string(json, &value, &unk_token_);
    }
    // Sanity floor for a loaded Qwen vocabulary (the 2B model has ~152k).
    constexpr size_t kMinVocabSize = 150000;
    if (token_to_id_.size() < kMinVocabSize || merge_rank_.empty()) {
        *error = "incomplete Qwen BPE vocabulary";
        return false;
    }
    return true;
}

bool QwenTokenizer::bpe(const std::string &piece, std::vector<int32_t> *ids,
                        std::string *error) const {
    std::vector<std::string> symbols;
    const std::string encoded = byte_encode(piece);
    for (size_t offset = 0; offset < encoded.size();) {
        uint32_t cp = 0;
        size_t width = 0;
        if (!utf8_at(encoded, offset, &cp, &width)) {
            *error = "invalid UTF-8 input";
            return false;
        }
        symbols.push_back(encoded.substr(offset, width));
        offset += width;
    }
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it =
                merge_rank_.find(symbols[i] + '\x1f' + symbols[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == symbols.size())
            break;
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() &&
                symbols[i] + '\x1f' + symbols[i + 1] ==
                    symbols[best] + '\x1f' + symbols[best + 1]) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(symbols[i++]);
            }
        }
        symbols.swap(merged);
    }
    for (const std::string &symbol : symbols) {
        const auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end()) {
            const auto unknown = token_to_id_.find(unk_token_);
            if (unknown == token_to_id_.end()) {
                *error = "BPE symbol absent from vocabulary";
                return false;
            }
            ids->push_back(unknown->second);
        } else
            ids->push_back(it->second);
    }
    return true;
}

bool QwenTokenizer::encode_regular(const std::string &text,
                                   std::vector<int32_t> *ids,
                                   std::string *error) const {
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        size_t width = 0;
        if (!utf8_at(text, offset, &cp, &width)) {
            *error = "invalid UTF-8 text";
            return false;
        }
        const size_t begin = offset;
        if (is_space(cp)) {
            // Qwen's pre-tokenizer attaches one optional non-letter/number
            // character (usually a space) to the next word or punctuation run.
            size_t after_space = offset + width;
            uint32_t next = 0;
            size_t next_width = 0;
            const bool has_next =
                utf8_at(text, after_space, &next, &next_width);
            if (cp == ' ' && has_next && is_letter(next)) {
                offset = after_space + next_width;
                while (offset < text.size()) {
                    uint32_t more = 0;
                    size_t more_width = 0;
                    utf8_at(text, offset, &more, &more_width);
                    if (!is_letter(more))
                        break;
                    offset += more_width;
                }
            } else if (cp == ' ' && has_next && !is_space(next) &&
                       !is_letter(next) && !is_digit(next)) {
                offset = after_space + next_width;
                while (offset < text.size()) {
                    uint32_t more = 0;
                    size_t more_width = 0;
                    utf8_at(text, offset, &more, &more_width);
                    if (is_space(more) || is_letter(more) || is_digit(more))
                        break;
                    offset += more_width;
                }
                while (offset < text.size()) {
                    uint32_t more = 0;
                    size_t more_width = 0;
                    utf8_at(text, offset, &more, &more_width);
                    if (!is_newline(more))
                        break;
                    offset += more_width;
                }
            } else {
                offset += width;
                while (offset < text.size()) {
                    uint32_t more = 0;
                    size_t more_width = 0;
                    utf8_at(text, offset, &more, &more_width);
                    if (!is_space(more))
                        break;
                    offset += more_width;
                }
            }
        } else if (is_letter(cp)) {
            offset += width;
            while (offset < text.size()) {
                uint32_t next = 0;
                size_t next_width = 0;
                utf8_at(text, offset, &next, &next_width);
                if (!is_letter(next))
                    break;
                offset += next_width;
            }
        } else if (is_digit(cp))
            offset += width;
        else {
            offset += width;
            while (offset < text.size()) {
                uint32_t next = 0;
                size_t next_width = 0;
                utf8_at(text, offset, &next, &next_width);
                if (is_space(next) || is_letter(next) || is_digit(next))
                    break;
                offset += next_width;
            }
        }
        if (!bpe(text.substr(begin, offset - begin), ids, error))
            return false;
    }
    return true;
}

bool QwenTokenizer::encode(const std::string &text, std::vector<int32_t> *ids,
                           std::string *error) const {
    if (ids == nullptr) {
        *error = "null token output";
        return false;
    }
    ids->clear();
    size_t offset = 0;
    size_t ordinary_begin = 0;
    while (offset < text.size()) {
        const auto found = std::find_if(
            added_tokens_.begin(), added_tokens_.end(), [&](const auto &item) {
                return text.compare(offset, item.first.size(), item.first) == 0;
            });
        if (found == added_tokens_.end()) {
            ++offset;
            continue;
        }
        if (offset > ordinary_begin &&
            !encode_regular(
                text.substr(ordinary_begin, offset - ordinary_begin), ids,
                error))
            return false;
        ids->push_back(found->second);
        offset += found->first.size();
        ordinary_begin = offset;
    }
    return ordinary_begin == text.size() ||
           encode_regular(text.substr(ordinary_begin), ids, error);
}

bool QwenTokenizer::decode(const std::vector<int32_t> &ids, std::string *text,
                           std::string *error) const {
    if (text == nullptr) {
        *error = "null text output";
        return false;
    }
    text->clear();
    std::string bytes;
    for (int32_t id : ids) {
        const auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            *error = "token id is absent from vocabulary";
            return false;
        }
        bytes += it->second;
    }
    *text = byte_decode(bytes);
    return true;
}

int32_t QwenTokenizer::token_id(const std::string &token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool QwenTokenizer::encode_chat(const std::vector<ChatMessage> &messages,
                                bool add_generation_prompt,
                                int32_t vision_token_count,
                                std::vector<int32_t> *ids,
                                std::string *error) const {
    if (vision_token_count < 0 || ids == nullptr) {
        *error = "invalid chat token output or vision token count";
        return false;
    }
    const int32_t im_start = token_id("<|im_start|>"),
                  im_end = token_id(kImEndToken),
                  vision_start = token_id("<|vision_start|>"),
                  vision_end = token_id("<|vision_end|>"),
                  image_pad = token_id(kImagePadToken);
    if (im_start < 0 || im_end < 0 || vision_start < 0 || vision_end < 0 ||
        image_pad < 0) {
        *error = "required Qwen special tokens are absent";
        return false;
    }
    ids->clear();
    for (const ChatMessage &message : messages) {
        if (message.role != "system" && message.role != "user" &&
            message.role != "assistant") {
            *error = "unsupported chat role";
            return false;
        }
        ids->push_back(im_start);
        if (!encode_regular(message.role + "\n", ids, error))
            return false;
        if (message.has_image) {
            ids->push_back(vision_start);
            ids->insert(ids->end(), static_cast<size_t>(vision_token_count),
                        image_pad);
            ids->push_back(vision_end);
        }
        std::vector<int32_t> content_ids;
        if (!encode(message.content, &content_ids, error))
            return false;
        ids->insert(ids->end(), content_ids.begin(), content_ids.end());
        ids->push_back(im_end);
        if (!encode_regular("\n", ids, error))
            return false;
    }
    if (add_generation_prompt) {
        ids->push_back(im_start);
        return encode_regular("assistant\n", ids, error);
    }
    return true;
}

} // namespace wevision
