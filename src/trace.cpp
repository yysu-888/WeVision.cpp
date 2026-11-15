#include "trace.hpp"

#include "runtime.hpp"

#include "ggml-backend.h"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace wevision {
namespace {

std::string sanitize_filename(const std::string &value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char ch : value) {
        result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
    }
    return result.empty() ? "tensor" : result;
}

std::string json_quote(const std::string &value) {
    std::string result = "\"";
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            result += "\\\\";
            break;
        case '\"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result += ch;
            break;
        }
    }
    result += "\"";
    return result;
}

uint64_t fnv1a64(const std::vector<uint8_t> &bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (const uint8_t value : bytes) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

FileTraceSink::FileTraceSink(std::filesystem::path root,
                             std::vector<std::string> selectors)
    : root_(std::move(root)), selectors_(std::move(selectors)) {}

bool FileTraceSink::wants(const TraceTensorInfo &info) const {
    if (selectors_.empty())
        return true;
    for (const std::string &selector : selectors_) {
        if (selector == info.name)
            return true;
        if (!selector.empty() && selector.back() == '*' &&
            info.name.compare(0, selector.size() - 1, selector, 0,
                              selector.size() - 1) == 0)
            return true;
    }
    return false;
}

bool FileTraceSink::checkpoint(TraceTensorInfo info, ggml_tensor *tensor,
                               std::string *error) {
    if (captured_) {
        *error = "cannot add a trace watch after capture";
        return false;
    }
    if (tensor == nullptr || info.name.empty()) {
        *error = "trace watch requires a name and tensor";
        return false;
    }
    for (const Watch &watch : watches_) {
        if (watch.info.name == info.name) {
            *error = "duplicate trace tensor name: " + info.name;
            return false;
        }
    }
    watches_.push_back({std::move(info), tensor});
    return true;
}

bool FileTraceSink::watch(TraceTensorInfo info, ggml_tensor *tensor,
                          std::string *error) {
    return checkpoint(std::move(info), tensor, error);
}

bool FileTraceSink::capture(const BackendContext &backend, std::string *error) {
    if (captured_) {
        *error = "trace sink has already captured this graph";
        return false;
    }
    if (!backend.ready()) {
        *error = "cannot capture from an unrealized backend";
        return false;
    }
    std::error_code filesystem_error;
    if (std::filesystem::exists(root_, filesystem_error) &&
        !std::filesystem::is_empty(root_, filesystem_error)) {
        *error = "trace output directory is not empty: " + root_.string();
        return false;
    }
    std::filesystem::create_directories(root_, filesystem_error);
    if (filesystem_error) {
        *error = "cannot create trace output directory: " + root_.string();
        return false;
    }

    struct Record {
        const Watch *watch = nullptr;
        std::string file;
        std::vector<uint8_t> bytes;
    };

    std::vector<Record> records;
    records.reserve(watches_.size());
    for (size_t index = 0; index < watches_.size(); ++index) {
        const Watch &watch = watches_[index];
        Record record;
        record.watch = &watch;
        record.file = std::to_string(index) + "_" +
                      sanitize_filename(watch.info.name) + ".bin";
        record.bytes.resize(ggml_nbytes(watch.tensor));
        ggml_backend_tensor_get(watch.tensor, record.bytes.data(), 0,
                                record.bytes.size());
        std::ofstream output(root_ / record.file, std::ios::binary);
        if (!output) {
            *error = "cannot write trace tensor: " + record.file;
            return false;
        }
        output.write(reinterpret_cast<const char *>(record.bytes.data()),
                     static_cast<std::streamsize>(record.bytes.size()));
        if (!output) {
            *error = "cannot finish trace tensor: " + record.file;
            return false;
        }
        records.push_back(std::move(record));
    }

    std::ofstream manifest(root_ / "manifest.json");
    if (!manifest) {
        *error = "cannot write trace manifest";
        return false;
    }
    manifest << "{\n  \"format_version\": 1,\n  \"hash\": \"fnv1a64\",\n  "
                "\"tensors\": {";
    for (size_t index = 0; index < records.size(); ++index) {
        const Record &record = records[index];
        const ggml_tensor *tensor = record.watch->tensor;
        if (index != 0)
            manifest << ',';
        manifest << "\n    " << json_quote(record.watch->info.name) << ": {"
                 << "\"file\": " << json_quote(record.file)
                 << ", \"dtype\": " << json_quote(ggml_type_name(tensor->type))
                 << ", \"layout\": " << json_quote(record.watch->info.layout)
                 << ", \"stage\": " << json_quote(record.watch->info.stage)
                 << ", \"shape\": [";
        for (int dim = ggml_n_dims(tensor) - 1; dim >= 0; --dim) {
            if (dim != ggml_n_dims(tensor) - 1)
                manifest << ", ";
            manifest << tensor->ne[dim];
        }
        manifest << "], \"bytes\": " << record.bytes.size()
                 << ", \"fnv1a64\": \"0x" << std::hex << std::setw(16)
                 << std::setfill('0') << fnv1a64(record.bytes) << std::dec
                 << "\"}";
    }
    manifest << "\n  }\n}\n";
    if (!manifest) {
        *error = "cannot finish trace manifest";
        return false;
    }
    captured_ = true;
    return true;
}

} // namespace wevision
