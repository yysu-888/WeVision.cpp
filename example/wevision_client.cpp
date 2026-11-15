// HTTP client example for wevision-serve. Sends a POST /infer request and
// prints the response in the same format as wevision-cli, optionally decoding
// base64 mask/alpha planes back to PNG files.
//
// Usage: wevision-client --url http://127.0.0.1:8717 \
//                        --image images/seabreeze.jpg --task detect \
//                        [--query "a person"] [--prompt "..."] \
//                        [--max-tokens 256] [--save-prefix outputs/cli]
//
// Equivalent curl calls:
//   curl -s http://127.0.0.1:8717/health
//   curl -s -X POST http://127.0.0.1:8717/infer \
//        -H 'Content-Type: application/json' \
//        -d '{"image_path":"images/seabreeze.jpg","task":"detect","query":"person"}'
//   curl -s -X POST http://127.0.0.1:8717/infer \
//        -H 'Content-Type: application/json' \
//        -d '{"image_path":"images/seabreeze.jpg","task":"matting","query":"person"}' \
//        -o matting.json
#include "httplib.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace {

using nlohmann::json;

struct Options {
    std::string url = "http://127.0.0.1:8717";
    std::string image;
    std::string task = "detect";
    std::string query;
    std::string prompt;
    std::string save_prefix;
    uint32_t max_tokens = 0;
};

void usage(const char *program) {
    std::fprintf(stderr,
                 "Usage: %s --url http://HOST:PORT --image PATH "
                 "--task detect|point|count|ocr|segment|matting "
                 "[--query TEXT] [--prompt TEXT] [--max-tokens N] "
                 "[--save-prefix PREFIX]\n",
                 program);
}

bool parse_options(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const char *value = index + 1 < argc ? argv[index + 1] : nullptr;
        if (flag == "--url" && value) {
            options->url = value;
            ++index;
        } else if (flag == "--image" && value) {
            options->image = value;
            ++index;
        } else if (flag == "--task" && value) {
            options->task = value;
            ++index;
        } else if (flag == "--query" && value) {
            options->query = value;
            ++index;
        } else if (flag == "--prompt" && value) {
            options->prompt = value;
            ++index;
        } else if (flag == "--save-prefix" && value) {
            options->save_prefix = value;
            ++index;
        } else if (flag == "--max-tokens" && value) {
            char *end = nullptr;
            const long tokens = std::strtol(value, &end, 10);
            if (value == end || *end != '\0' || tokens <= 0 || tokens > 65536)
                return false;
            options->max_tokens = static_cast<uint32_t>(tokens);
            ++index;
        } else {
            return false;
        }
    }
    return !options->image.empty();
}

json build_request_body(const Options &options) {
    json body;
    body["image_path"] = options.image;
    body["task"] = options.task;
    if (!options.query.empty())
        body["query"] = options.query;
    if (!options.prompt.empty())
        body["prompt"] = options.prompt;
    if (options.max_tokens != 0)
        body["max_tokens"] = options.max_tokens;
    return body;
}

int base64_value(char c) {
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

bool base64_decode(const std::string &text, std::string *out) {
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
        if (padding == 2 && index + 4 != text.size())
            return false;
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

bool save_plane(const Options &options, const json &box, const char *key,
                const char *suffix, uint32_t index) {
    if (!box.contains(key) || !box[key].is_string())
        return true; // task without planes
    std::string png;
    if (!base64_decode(box[key].get<std::string>(), &png)) {
        std::fprintf(stderr, "base64 decode failed for %s[%u]\n", suffix,
                     index);
        return false;
    }
    const std::string path = options.save_prefix + "-" + suffix + "-" +
                             std::to_string(index) + ".png";
    std::ofstream out(path, std::ios::binary);
    if (!out || !out.write(png.data(),
                           static_cast<std::streamsize>(png.size()))) {
        std::fprintf(stderr, "write failed: %s\n", path.c_str());
        return false;
    }
    std::printf("%s[%u]=%s\n", suffix, index, path.c_str());
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }

    httplib::Client client(options.url);
    client.set_connection_timeout(30, 0);
    client.set_read_timeout(300, 0);

    const std::string body = build_request_body(options).dump();
    httplib::Result result = client.Post("/infer", body, "application/json");
    if (!result) {
        std::fprintf(stderr, "request failed: %s\n",
                     httplib::to_string(result.error()).c_str());
        return 1;
    }
    if (result->status != 200) {
        json error_body;
        try {
            error_body = json::parse(result->body);
        } catch (const std::exception &) {
            error_body = json();
        }
        if (error_body.contains("error") &&
            error_body["error"].is_string()) {
            std::fprintf(stderr, "HTTP %d: %s\n", result->status,
                         error_body["error"].get<std::string>().c_str());
            return 1;
        }
        std::fprintf(stderr, "HTTP %d: %s\n", result->status,
                     result->body.c_str());
        return 1;
    }

    json response;
    try {
        response = json::parse(result->body);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "response JSON parse failed: %s\n",
                     error.what());
        return 1;
    }

    if (response.contains("text") && response["text"].is_string())
        std::printf("text=%s\n", response["text"].get<std::string>().c_str());
    if (response.contains("count") && response["count"].is_number())
        std::printf("count=%d\n",
                    response["count"].get<int>());

    if (response.contains("boxes") && response["boxes"].is_array()) {
        const json boxes = response["boxes"];
        for (uint32_t index = 0; index < boxes.size(); ++index) {
            const json &box = boxes[index];
            const auto coordinate = [&box](const char *key) {
                return (box.contains(key) && box[key].is_number())
                           ? box[key].get<double>()
                           : 0.0;
            };
            std::string label;
            if (box.contains("label") && box["label"].is_string())
                label = box["label"];
            std::printf("box[%u]=%.2f,%.2f,%.2f,%.2f label=%s\n", index,
                        coordinate("x1"), coordinate("y1"),
                        coordinate("x2"), coordinate("y2"),
                        label.c_str());
            if (!options.save_prefix.empty()) {
                if (!save_plane(options, box, "mask_png_b64", "mask",
                                index) ||
                    !save_plane(options, box, "alpha_png_b64", "alpha",
                                index))
                    return 1;
            }
        }
    }

    if (response.contains("points") && response["points"].is_array()) {
        const json points = response["points"];
        for (uint32_t index = 0; index < points.size(); ++index) {
            const json &point = points[index];
            const auto coordinate = [&point](const char *key) {
                return (point.contains(key) && point[key].is_number())
                           ? point[key].get<double>()
                           : 0.0;
            };
            std::string label;
            if (point.contains("label") && point["label"].is_string())
                label = point["label"];
            std::printf("point[%u]=%.2f,%.2f label=%s\n", index,
                        coordinate("x"), coordinate("y"), label.c_str());
        }
    }

    if (response.contains("timings_ms") &&
        response["timings_ms"].is_object()) {
        std::printf("timings_ms");
        for (const auto &entry : response["timings_ms"].items()) {
            if (entry.value().is_number())
                std::printf(" %s=%.2f", entry.key().c_str(),
                             entry.value().get<double>());
        }
        std::printf("\n");
    }
    if (response.contains("wall_ms") && response["wall_ms"].is_number())
        std::printf("wall_ms=%.2f\n", response["wall_ms"].get<double>());
    return 0;
}
