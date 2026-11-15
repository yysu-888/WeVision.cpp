// Minimal self-contained JSON and base64 helpers shared by the HTTP serve and
// client examples. Only what the /infer protocol needs: parsing flat request
// objects, escaping response strings, and embedding PNG planes as base64.
#ifndef WEVISION_HTTP_JSON_HPP
#define WEVISION_HTTP_JSON_HPP

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace wevision_http {

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonValue> items;
    std::vector<std::pair<std::string, JsonValue>> fields;

    const JsonValue *find(const char *key) const {
        if (type != Object)
            return nullptr;
        for (const auto &entry : fields)
            if (entry.first == key)
                return &entry.second;
        return nullptr;
    }
};

inline void append_utf8(std::string *out, uint32_t code) {
    if (code < 0x80) {
        out->push_back(static_cast<char>(code));
    } else if (code < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (code >> 6)));
        out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (code >> 12)));
        out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (code >> 18)));
        out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
}

struct JsonCursor {
    const char *p;
    const char *end;
    std::string error;

    void skip() {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }
    bool fail(const char *message) {
        if (error.empty())
            error = message;
        return false;
    }
    bool parse_hex4(uint32_t *out) {
        if (end - p < 4)
            return false;
        uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const char c = p[index];
            value <<= 4;
            if (c >= '0' && c <= '9')
                value |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                value |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                value |= static_cast<uint32_t>(c - 'A' + 10);
            else
                return false;
        }
        p += 4;
        *out = value;
        return true;
    }
    bool parse_string(std::string *out) {
        if (p == end || *p != '"')
            return fail("expected string");
        ++p;
        while (p < end && *p != '"') {
            const char c = *p++;
            if (static_cast<unsigned char>(c) < 0x20)
                return fail("control character in string");
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (p == end)
                return fail("truncated escape");
            const char e = *p++;
            switch (e) {
            case '"': out->push_back('"'); break;
            case '\\': out->push_back('\\'); break;
            case '/': out->push_back('/'); break;
            case 'b': out->push_back('\b'); break;
            case 'f': out->push_back('\f'); break;
            case 'n': out->push_back('\n'); break;
            case 'r': out->push_back('\r'); break;
            case 't': out->push_back('\t'); break;
            case 'u': {
                uint32_t code = 0;
                if (!parse_hex4(&code))
                    return fail("bad \\u escape");
                if (code >= 0xD800 && code <= 0xDBFF && end - p >= 6 &&
                    p[0] == '\\' && p[1] == 'u') {
                    p += 2;
                    uint32_t low = 0;
                    if (!parse_hex4(&low) || low < 0xDC00 || low > 0xDFFF)
                        return fail("bad surrogate pair");
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                }
                append_utf8(out, code);
                break;
            }
            default:
                return fail("unknown escape");
            }
        }
        if (p == end)
            return fail("unterminated string");
        ++p;
        return true;
    }
    bool parse_value(JsonValue *out) {
        skip();
        if (p == end)
            return fail("unexpected end");
        const char c = *p;
        if (c == '"') {
            out->type = JsonValue::String;
            return parse_string(&out->text);
        }
        if (c == '{') {
            ++p;
            out->type = JsonValue::Object;
            skip();
            if (p < end && *p == '}') {
                ++p;
                return true;
            }
            while (true) {
                skip();
                std::string key;
                if (!parse_string(&key))
                    return false;
                skip();
                if (p == end || *p != ':')
                    return fail("expected ':'");
                ++p;
                JsonValue value;
                if (!parse_value(&value))
                    return false;
                out->fields.emplace_back(std::move(key), std::move(value));
                skip();
                if (p < end && *p == ',') {
                    ++p;
                    continue;
                }
                if (p < end && *p == '}') {
                    ++p;
                    return true;
                }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            out->type = JsonValue::Array;
            skip();
            if (p < end && *p == ']') {
                ++p;
                return true;
            }
            while (true) {
                JsonValue value;
                if (!parse_value(&value))
                    return false;
                out->items.push_back(std::move(value));
                skip();
                if (p < end && *p == ',') {
                    ++p;
                    continue;
                }
                if (p < end && *p == ']') {
                    ++p;
                    return true;
                }
                return fail("expected ',' or ']'");
            }
        }
        if (end - p >= 4 && std::strncmp(p, "true", 4) == 0) {
            p += 4;
            out->type = JsonValue::Bool;
            out->boolean = true;
            return true;
        }
        if (end - p >= 5 && std::strncmp(p, "false", 5) == 0) {
            p += 5;
            out->type = JsonValue::Bool;
            out->boolean = false;
            return true;
        }
        if (end - p >= 4 && std::strncmp(p, "null", 4) == 0) {
            p += 4;
            return true;
        }
        char *stop = nullptr;
        const double value = std::strtod(p, &stop);
        if (stop == p)
            return fail("invalid literal");
        p = stop;
        out->type = JsonValue::Number;
        out->number = value;
        return true;
    }
};

inline bool json_parse(const std::string &input, JsonValue *out,
                       std::string *error) {
    JsonCursor cursor{input.data(), input.data() + input.size(), {}};
    if (!cursor.parse_value(out)) {
        if (error)
            *error = cursor.error.empty() ? "parse failed" : cursor.error;
        return false;
    }
    cursor.skip();
    if (cursor.p != cursor.end) {
        if (error)
            *error = "trailing characters after JSON value";
        return false;
    }
    return true;
}

inline std::string json_escape(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char raw : value) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            } else {
                out.push_back(raw);
            }
        }
    }
    return out;
}

inline std::string json_number(double value) {
    if (std::isfinite(value)) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.6g", value);
        return buffer;
    }
    return "null";
}

inline const char *base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline std::string base64_encode(const void *data, size_t size) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (size_t index = 0; index < size; index += 3) {
        const uint32_t b0 = bytes[index];
        const uint32_t b1 = index + 1 < size ? bytes[index + 1] : 0;
        const uint32_t b2 = index + 2 < size ? bytes[index + 2] : 0;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(base64_chars[(triple >> 18) & 0x3F]);
        out.push_back(base64_chars[(triple >> 12) & 0x3F]);
        out.push_back(index + 1 < size ? base64_chars[(triple >> 6) & 0x3F]
                                       : '=');
        out.push_back(index + 2 < size ? base64_chars[triple & 0x3F] : '=');
    }
    return out;
}

inline int base64_value(char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

inline bool base64_decode(const std::string &text, std::string *out) {
    if (text.size() % 4 != 0)
        return false;
    out->clear();
    out->reserve(text.size() / 4 * 3);
    for (size_t index = 0; index < text.size(); index += 4) {
        int v[4];
        int padding = 0;
        for (int slot = 0; slot < 4; ++slot) {
            const char c = text[index + static_cast<size_t>(slot)];
            if (c == '=') {
                ++padding;
                v[slot] = 0;
            } else {
                if (padding)
                    return false;
                v[slot] = base64_value(c);
                if (v[slot] < 0)
                    return false;
            }
        }
        if (padding == 2 || padding == 1) {
            if (padding == 2 && index + 4 != text.size())
                return false;
        }
        const uint32_t triple =
            (static_cast<uint32_t>(v[0]) << 18) |
            (static_cast<uint32_t>(v[1]) << 12) |
            (static_cast<uint32_t>(v[2]) << 6) | static_cast<uint32_t>(v[3]);
        out->push_back(static_cast<char>((triple >> 16) & 0xFF));
        if (padding < 2)
            out->push_back(static_cast<char>((triple >> 8) & 0xFF));
        if (padding < 1)
            out->push_back(static_cast<char>(triple & 0xFF));
    }
    return true;
}

} // namespace wevision_http

#endif // WEVISION_HTTP_JSON_HPP
