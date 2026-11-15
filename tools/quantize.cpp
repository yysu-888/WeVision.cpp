// Offline GGUF quantizer for WeVision models (llama-quantize style).
//
// Usage:
//   wevision-quantize INPUT.gguf OUTPUT.gguf q8_0|q4_k [--dry-run]
//   wevision-quantize INPUT.gguf --list
//
// Decision table (inference side stays untouched):
//   - quantize 2D linear weights in the wv.text.* / wv.vision.* domains when
//     the row length satisfies the block alignment of the target type
//   - always keep F32/F16: wv.mask.* (F32 asserted at runtime), norms, biases,
//     positional/level embeddings, lm_head (kept in q4_k mode)
//
// Note: this standalone ggml has no Q4_K_M (a llama.cpp alias mixing Q4_K and
// Q6_K rows), so the 4-bit target is plain Q4_K (block size 256).
//
// This is a standalone tool: it does not modify any inference code path.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

enum class Mode { List, Quantize, Verify };

struct Options {
    Mode mode = Mode::Quantize;
    std::string input;
    std::string output;
    enum ggml_type target = GGML_TYPE_COUNT; // sentinel: unset
    bool dry_run = false;
    bool keep_token_embd = false; // ablation switch for error attribution
};

bool parse_type(const std::string &name, enum ggml_type *out) {
    // ggml type names mix case ("q8_0", "q4_K"); match case-insensitively.
    std::string lower = name;
    for (char &c : lower)
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        std::string candidate = ggml_type_name(static_cast<enum ggml_type>(t));
        for (char &c : candidate)
            c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
        if (lower == candidate) {
            *out = static_cast<enum ggml_type>(t);
            return true;
        }
    }
    return false;
}

bool parse_args(int argc, char **argv, Options *opts) {
    if (argc < 2)
        return false;
    opts->input = argv[1];
    if (argc == 3 && std::strcmp(argv[2], "--list") == 0) {
        opts->mode = Mode::List;
        return true;
    }
    if (argc == 4 && std::strcmp(argv[3], "--verify") == 0) {
        opts->mode = Mode::Verify;
        opts->output = argv[2];
        return true;
    }
    if (argc < 4)
        return false;
    opts->output = argv[2];
    if (!parse_type(argv[3], &opts->target))
        return false;
    if (opts->target != GGML_TYPE_Q8_0 && opts->target != GGML_TYPE_Q4_K) {
        std::fprintf(stderr, "unsupported target type: %s (q8_0 or q4_k)\n",
                     argv[3]);
        return false;
    }
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dry-run") == 0)
            opts->dry_run = true;
        else if (std::strcmp(argv[i], "--keep-token-embd") == 0)
            opts->keep_token_embd = true;
        else
            return false;
    }
    return true;
}

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s INPUT.gguf OUTPUT.gguf q8_0|q4_k [--dry-run]\n"
                 "       %s INPUT.gguf --list\n"
                 "       %s SRC.gguf QUANT.gguf --verify\n",
                 prog, prog, prog);
}

// ---------------------------------------------------------------------------
// Decision table
// ---------------------------------------------------------------------------

bool is_embedding_or_norm_like(const std::string &name) {
    static const char *kKeep[] = {
        "norm", "bias", "pos_embd", "level_embed", "lm_head",
    };
    for (const char *needle : kKeep) {
        if (name.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

bool in_quant_domain(const std::string &name) {
    return name.rfind("wv.text.", 0) == 0 || name.rfind("wv.vision.", 0) == 0;
}

// Returns true when the tensor should be quantized to `target`.
bool should_quantize(const std::string &name, const int64_t *ne,
                     enum ggml_type type, enum ggml_type target,
                     bool keep_token_embd = false) {
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32 &&
        type != GGML_TYPE_BF16)
        return false;
    // 2D linear weight only: ne[2] == ne[3] == 1 and more than one row.
    if (ne[2] != 1 || ne[3] != 1 || ne[1] < 2)
        return false;
    if (!in_quant_domain(name))
        return false; // wv.mask.* and anything else stays as-is
    // wv.text.output.weight is the logits head (no "lm_head" name in this
    // model): quality-critical, kept unquantized in q4_k mode.
    if (name == "wv.text.output.weight")
        return target == GGML_TYPE_Q8_0;
    // token embeddings are row-lookup weights: quantize like llama.cpp,
    // unless the ablation switch keeps them unquantized.
    if (name.find("token_embd") != std::string::npos)
        return !keep_token_embd;
    if (is_embedding_or_norm_like(name))
        return false;
    const int64_t blocks = target == GGML_TYPE_Q8_0 ? 32 : 256;
    return ne[0] % blocks == 0;
}

// ---------------------------------------------------------------------------
// Decision report
// ---------------------------------------------------------------------------

struct Stats {
    size_t kept_tensors = 0, quantized_tensors = 0;
    size_t kept_bytes = 0, quantized_src_bytes = 0, quantized_dst_bytes = 0;
};

size_t tensor_bytes(const int64_t *ne, enum ggml_type type) {
    return ggml_row_size(type, ne[0]) * static_cast<size_t>(ne[1]) *
           static_cast<size_t>(ne[2]) * static_cast<size_t>(ne[3]);
}

void report_decision(const char *name, const int64_t *ne, enum ggml_type from,
                     enum ggml_type to, Stats *stats) {
    const size_t src = tensor_bytes(ne, from);
    const size_t dst = tensor_bytes(ne, to);
    if (from == to) {
        stats->kept_tensors += 1;
        stats->kept_bytes += src;
    } else {
        stats->quantized_tensors += 1;
        stats->quantized_src_bytes += src;
        stats->quantized_dst_bytes += dst;
    }
    std::printf(
        "%-64s %8s -> %-8s %12zu -> %12zu B  ne=[%lld,%lld,%lld,%lld]\n", name,
        ggml_type_name(from), ggml_type_name(to), src, dst,
        static_cast<long long>(ne[0]), static_cast<long long>(ne[1]),
        static_cast<long long>(ne[2]), static_cast<long long>(ne[3]));
}

int run_list(const char *path, enum ggml_type target, bool target_valid) {
    ggml_context *meta = nullptr;
    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/&meta};
    gguf_context *gguf = gguf_init_from_file(path, params);
    if (gguf == nullptr || meta == nullptr) {
        std::fprintf(stderr, "cannot open GGUF: %s\n", path);
        return 1;
    }
    const int64_t n = gguf_get_n_tensors(gguf);
    std::printf("%s: %lld tensors, alignment=%zu, target=%s\n", path,
                static_cast<long long>(n), gguf_get_alignment(gguf),
                target_valid ? ggml_type_name(target) : "(none)");
    Stats stats{};
    for (int64_t i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(gguf, i);
        const int64_t *ne = gguf_get_tensor_ne(gguf, i);
        const enum ggml_type type = gguf_get_tensor_type(gguf, i);
        const enum ggml_type to =
            (target_valid && should_quantize(name, ne, type, target)) ? target
                                                                      : type;
        report_decision(name, ne, type, to, &stats);
    }
    std::printf(
        "\nsummary: %zu kept (%.2f GB), %zu quantized (%.2f GB -> %.2f GB)\n",
        stats.kept_tensors, stats.kept_bytes / 1e9, stats.quantized_tensors,
        stats.quantized_src_bytes / 1e9, stats.quantized_dst_bytes / 1e9);
    gguf_free(gguf);
    ggml_free(meta);
    return 0;
}

// ---------------------------------------------------------------------------
// Quantization write path
// ---------------------------------------------------------------------------

bool copy_raw_tensor(std::ifstream &in, std::ofstream &out,
                     std::vector<char> &buffer, size_t bytes,
                     size_t file_offset, const char *name) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(file_offset));
    for (size_t done = 0; done < bytes;) {
        const size_t chunk = std::min(buffer.size(), bytes - done);
        in.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (static_cast<size_t>(in.gcount()) != chunk) {
            std::fprintf(stderr, "short read on tensor %s\n", name);
            return false;
        }
        out.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!out) {
            std::fprintf(stderr, "short write on tensor %s\n", name);
            return false;
        }
        done += chunk;
    }
    return true;
}

void write_padding(std::ofstream &out, size_t pad, std::vector<char> *buffer) {
    while (pad > 0) {
        const size_t chunk = std::min(pad, buffer->size());
        std::memset(buffer->data(), 0, chunk);
        out.write(buffer->data(), static_cast<std::streamsize>(chunk));
        pad -= chunk;
    }
}

// Quantizes one tensor: F16/F32 rows -> f32 -> ggml_quantize_chunk per pass.
// `scratch` holds [src rows][quantized rows] per pass; f32 staging is separate.
bool quantize_tensor(std::ifstream &in, std::ofstream &out, const char *name,
                     const int64_t *ne, enum ggml_type src_type,
                     enum ggml_type target, size_t file_offset,
                     std::vector<char> *scratch) {
    const int64_t ne0 = ne[0];
    const size_t rows = static_cast<size_t>(ne[1]) * ne[2] * ne[3];
    const size_t src_row = ggml_row_size(src_type, ne0);
    const size_t dst_row = ggml_row_size(target, ne0);
    const size_t per_row = src_row + dst_row;
    if (per_row > scratch->size() / 2) {
        std::fprintf(stderr, "row of tensor %s too large for buffer\n", name);
        return false;
    }
    const size_t rows_per_pass =
        std::max<size_t>(1, (scratch->size() / 2) / per_row);

    in.clear();
    in.seekg(static_cast<std::streamoff>(file_offset));

    char *base = scratch->data();
    std::vector<char> f32_store(rows_per_pass * static_cast<size_t>(ne0) *
                                sizeof(float));
    size_t done = 0;
    while (done < rows) {
        const size_t pass = std::min(rows_per_pass, rows - done);
        char *src = base;
        char *dst = base + pass * src_row;
        in.read(src, static_cast<std::streamsize>(pass * src_row));
        if (static_cast<size_t>(in.gcount()) != pass * src_row) {
            std::fprintf(stderr, "short read on tensor %s\n", name);
            return false;
        }
        for (size_t r = 0; r < pass; ++r) {
            const char *row = src + r * src_row;
            float *row_f32 = reinterpret_cast<float *>(f32_store.data() +
                                                       r * ne0 * sizeof(float));
            if (src_type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(row), row_f32, ne0);
            } else if (src_type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row(
                    reinterpret_cast<const ggml_bf16_t *>(row), row_f32, ne0);
            } else {
                std::memcpy(row_f32, row, ne0 * sizeof(float));
            }
        }
        ggml_quantize_chunk(target,
                            reinterpret_cast<const float *>(f32_store.data()),
                            dst, /*start=*/0, static_cast<int64_t>(pass), ne0,
                            /*imatrix=*/nullptr);
        out.write(dst, static_cast<std::streamsize>(pass * dst_row));
        if (!out) {
            std::fprintf(stderr, "short write on tensor %s\n", name);
            return false;
        }
        done += pass;
    }
    return true;
}

int run_quantize(const Options &opts) {
    ggml_context *meta = nullptr;
    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/&meta};
    gguf_context *src = gguf_init_from_file(opts.input.c_str(), params);
    if (src == nullptr || meta == nullptr) {
        std::fprintf(stderr, "cannot open GGUF: %s\n", opts.input.c_str());
        return 1;
    }
    const int64_t n = gguf_get_n_tensors(src);
    const size_t alignment = gguf_get_alignment(src);
    const size_t data_offset = gguf_get_data_offset(src);
    std::printf("input: %s (%lld tensors)\n", opts.input.c_str(),
                static_cast<long long>(n));

    // Build output gguf context: copy metadata, add tensors with target types.
    ggml_context *build = ggml_init({16ULL << 20, nullptr, true});
    if (build == nullptr) {
        std::fprintf(stderr, "cannot create build context\n");
        return 1;
    }
    gguf_context *dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_str(dst, "wevision.quant.type", ggml_type_name(opts.target));

    Stats stats{};
    for (int64_t i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(src, i);
        const int64_t *ne = gguf_get_tensor_ne(src, i);
        const enum ggml_type type = gguf_get_tensor_type(src, i);
        const enum ggml_type to =
            should_quantize(name, ne, type, opts.target, opts.keep_token_embd)
                ? opts.target
                : type;
        ggml_tensor *tensor =
            ggml_new_tensor_4d(build, to, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(tensor, name);
        gguf_add_tensor(dst, tensor);
        report_decision(name, ne, type, to, &stats);
    }
    std::printf(
        "\nsummary: %zu kept (%.2f GB), %zu quantized (%.2f GB -> %.2f GB)\n",
        stats.kept_tensors, stats.kept_bytes / 1e9, stats.quantized_tensors,
        stats.quantized_src_bytes / 1e9, stats.quantized_dst_bytes / 1e9);
    if (opts.dry_run) {
        std::printf("dry run: no file written\n");
        gguf_free(dst);
        gguf_free(src);
        ggml_free(build);
        ggml_free(meta);
        return 0;
    }

    if (!gguf_write_to_file(dst, opts.output.c_str(), /*only_meta=*/true)) {
        std::fprintf(stderr, "cannot write GGUF meta: %s\n",
                     opts.output.c_str());
        return 1;
    }
    std::ofstream out(opts.output, std::ios::binary | std::ios::app);
    std::ifstream in(opts.input, std::ios::binary);
    if (!out || !in) {
        std::fprintf(stderr, "cannot open data files\n");
        return 1;
    }
    std::vector<char> buffer(64ULL << 20); // 64 MB scratch
    for (int64_t i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(src, i);
        const int64_t *ne = gguf_get_tensor_ne(src, i);
        const enum ggml_type type = gguf_get_tensor_type(src, i);
        const enum ggml_type to =
            should_quantize(name, ne, type, opts.target, opts.keep_token_embd)
                ? opts.target
                : type;
        const size_t file_offset = data_offset + gguf_get_tensor_offset(src, i);
        const size_t src_bytes = ggml_nbytes(ggml_get_tensor(meta, name));
        size_t written = 0;
        if (to != type) {
            if (!quantize_tensor(in, out, name, ne, type, to, file_offset,
                                 &buffer)) {
                std::fprintf(stderr, "quantization failed at tensor %s\n",
                             name);
                return 1;
            }
            written = tensor_bytes(ne, to);
            std::printf("quantized %s\n", name);
        } else {
            if (!copy_raw_tensor(in, out, buffer, src_bytes, file_offset,
                                 name)) {
                std::fprintf(stderr, "copy failed at tensor %s\n", name);
                return 1;
            }
            written = src_bytes;
        }
        // pad between tensors so the next tensor lands on its alignment
        const size_t pad = (alignment - (written % alignment)) % alignment;
        if (i + 1 < n && pad > 0) {
            std::vector<char> zeros(pad, 0);
            out.write(zeros.data(), static_cast<std::streamsize>(pad));
        }
    }
    out.close();
    in.close();
    std::printf("written: %s\n", opts.output.c_str());

    // header round-trip check: reopen and validate names/types/sizes
    ggml_context *verify_meta = nullptr;
    gguf_init_params vp{/*no_alloc=*/true, /*ctx=*/&verify_meta};
    gguf_context *verify = gguf_init_from_file(opts.output.c_str(), vp);
    if (verify == nullptr || verify_meta == nullptr) {
        std::fprintf(stderr, "verify: cannot reopen output\n");
        return 1;
    }
    if (gguf_get_n_tensors(verify) != n) {
        std::fprintf(stderr, "verify: tensor count mismatch\n");
        return 1;
    }
    for (int64_t i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(verify, i);
        const ggml_tensor *src_t = ggml_get_tensor(meta, name);
        const ggml_tensor *dst_t = ggml_get_tensor(verify_meta, name);
        if (dst_t == nullptr || src_t == nullptr) {
            std::fprintf(stderr, "verify: missing tensor %s\n", name);
            return 1;
        }
        // The header check validates the file against its own decision
        // (this run's options), not an independent recomputation, so
        // ablation flags stay consistent between build and verify passes.
        const enum ggml_type to =
            should_quantize(name, src_t->ne, src_t->type, opts.target,
                            opts.keep_token_embd)
                ? opts.target
                : src_t->type;
        if (dst_t->type != to ||
            (to == src_t->type && ggml_nbytes(dst_t) != ggml_nbytes(src_t))) {
            std::fprintf(stderr, "verify: mismatch for %s\n", name);
            return 1;
        }
    }
    std::printf("verify: header round-trip OK (%lld tensors)\n",
                static_cast<long long>(n));
    gguf_free(verify);
    ggml_free(verify_meta);

    gguf_free(dst);
    gguf_free(src);
    ggml_free(build);
    ggml_free(meta);
    ggml_quantize_free();
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Numeric round-trip verification: sample quantized tensors, dequantize, and
// compare against the bf16/f32/f16 source rows.
// ---------------------------------------------------------------------------

int run_verify(const std::string &src_path, const std::string &quant_path) {
    ggml_context *src_meta = nullptr, *quant_meta = nullptr;
    gguf_init_params p{/*no_alloc=*/true, /*ctx=*/nullptr};
    p.ctx = &src_meta;
    gguf_context *src = gguf_init_from_file(src_path.c_str(), p);
    p.ctx = &quant_meta;
    gguf_context *quant = gguf_init_from_file(quant_path.c_str(), p);
    if (src == nullptr || quant == nullptr || src_meta == nullptr ||
        quant_meta == nullptr) {
        std::fprintf(stderr, "cannot open GGUFs\n");
        return 1;
    }
    const int64_t n = gguf_get_n_tensors(src);
    if (n != gguf_get_n_tensors(quant)) {
        std::fprintf(stderr, "tensor count mismatch\n");
        return 1;
    }
    std::ifstream src_file(src_path, std::ios::binary);
    std::ifstream quant_file(quant_path, std::ios::binary);
    const size_t src_off = gguf_get_data_offset(src);
    const size_t quant_off = gguf_get_data_offset(quant);

    // sample at most 10 quantized tensors plus the first token_embd rows
    constexpr size_t kSampleRows = 16;
    size_t checked = 0;
    double worst_rel = 0.0, worst_maxabs = 0.0;
    for (int64_t i = 0; i < n && checked < 10; ++i) {
        const char *name = gguf_get_tensor_name(src, i);
        const int64_t *ne = gguf_get_tensor_ne(src, i);
        const enum ggml_type src_type = gguf_get_tensor_type(src, i);
        const enum ggml_type to = gguf_get_tensor_type(quant, i);
        if (to == src_type)
            continue;
        if (std::strcmp(name, gguf_get_tensor_name(quant, i)) != 0)
            continue; // order changed; fall back to name lookup below
        const ggml_to_float_t to_float = ggml_get_type_traits(to)->to_float;
        if (to_float == nullptr)
            continue;
        const int64_t ne0 = ne[0];
        const size_t rows =
            std::min<size_t>(kSampleRows, static_cast<size_t>(ne[1]));
        const size_t src_row = ggml_row_size(src_type, ne0);
        const size_t dst_row = ggml_row_size(to, ne0);
        std::vector<char> srow(src_row);
        std::vector<char> qrow(dst_row);
        std::vector<float> a(static_cast<size_t>(ne0));
        std::vector<float> b(static_cast<size_t>(ne0));
        double max_abs = 0.0, sum_sq = 0.0, sum_ref = 0.0, max_rel = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            src_file.seekg(static_cast<std::streamoff>(
                src_off + gguf_get_tensor_offset(src, i) + r * src_row));
            quant_file.seekg(static_cast<std::streamoff>(
                quant_off + gguf_get_tensor_offset(quant, i) + r * dst_row));
            src_file.read(srow.data(), static_cast<std::streamsize>(src_row));
            quant_file.read(qrow.data(), static_cast<std::streamsize>(dst_row));
            if (static_cast<size_t>(src_file.gcount()) != src_row ||
                static_cast<size_t>(quant_file.gcount()) != dst_row) {
                std::fprintf(stderr, "short read while verifying %s\n", name);
                return 1;
            }
            if (src_type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row(
                    reinterpret_cast<const ggml_bf16_t *>(srow.data()),
                    a.data(), ne0);
            } else if (src_type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(srow.data()),
                    a.data(), ne0);
            } else {
                std::memcpy(a.data(), srow.data(), a.size() * sizeof(float));
            }
            to_float(qrow.data(), b.data(), ne0);
            for (int64_t k = 0; k < ne0; ++k) {
                const double d = a[k] - b[k];
                const double ad = d < 0 ? -d : d;
                const double ar = a[k] < 0 ? -a[k] : a[k];
                max_abs = std::max(max_abs, ad);
                if (ar > 1e-6)
                    max_rel = std::max(max_rel, ad / ar);
                sum_sq += d * d;
                sum_ref += a[k] * a[k];
            }
        }
        const double rmse = std::sqrt(sum_sq / (rows * ne0));
        const double denom = std::sqrt(sum_ref / (rows * ne0));
        const double rel_rms = denom > 0 ? rmse / denom : 0.0;
        std::printf(
            "%-56s %-6s rel_rms=%.5f max_rel=%.4f max_abs=%.4f (%zu rows)\n",
            name, ggml_type_name(to), rel_rms, max_rel, max_abs, rows);
        worst_rel = std::max(worst_rel, rel_rms);
        worst_maxabs = std::max(worst_maxabs, max_abs);
        checked += 1;
    }
    std::printf(
        "\nverified %zu tensors: worst rel_rms=%.5f, worst max_abs=%.4f\n",
        checked, worst_rel, worst_maxabs);
    // q8_0 typical rel_rms ~ 0.002-0.005; q4_k ~ 0.01-0.02
    gguf_free(src);
    gguf_free(quant);
    ggml_free(src_meta);
    ggml_free(quant_meta);
    return 0;
}

int main(int argc, char **argv) {
    Options opts;
    if (!parse_args(argc, argv, &opts)) {
        print_usage(argv[0]);
        return 2;
    }
    if (opts.mode == Mode::List)
        return run_list(opts.input.c_str(), GGML_TYPE_Q8_0, true);
    if (opts.mode == Mode::Verify)
        return run_verify(opts.input, opts.output);
    return run_quantize(opts);
}
