#include "profile.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace wevision {

namespace {

// Enabled unless unset, empty, or "0".
bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

// Enabled when present, regardless of value.
bool env_present(const char *name) { return std::getenv(name) != nullptr; }

} // namespace

DebugOptions DebugOptions::from_environment() {
    DebugOptions options;
    options.operation_profile = env_flag("WEVISION_OP_PROFILE");
    options.stage_profile_disabled =
        env_present("WEVISION_DISABLE_STAGE_PROFILE");
    options.mask_profile_details = env_present("WEVISION_MASK_PROFILE_DETAILS");
    options.mask_s1_f32_direct = env_present("WEVISION_MASK_S1_F32_DIRECT");
    return options;
}

void TimingReport::add(std::string_view name, double milliseconds) {
    entries_[std::string(name)] += milliseconds;
}

void TimingReport::merge(const TimingReport &other) {
    for (const auto &entry : other.entries_)
        entries_[entry.first] += entry.second;
}

double TimingReport::get(std::string_view name) const {
    const auto found = entries_.find(std::string(name));
    return found == entries_.end() ? 0.0 : found->second;
}

void TimingReport::print(FILE *out, std::string_view prefix) const {
    std::vector<std::pair<std::string, double>> sorted(entries_.begin(),
                                                       entries_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &left, const auto &right) {
                  return left.first < right.first;
              });
    for (const auto &entry : sorted)
        std::fprintf(out, "%s%s=%.3f\n", std::string(prefix).c_str(),
                     entry.first.c_str(), entry.second);
}

void TimingReport::clear() { entries_.clear(); }

bool TimingReport::empty() const { return entries_.empty(); }

ScopedTimer::ScopedTimer(double *sink)
    : sink_(sink), started_(std::chrono::steady_clock::now()) {}

ScopedTimer::ScopedTimer(std::string name, TimingReport &report)
    : name_(std::move(name)), report_(&report),
      started_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const double milliseconds = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started_)
                                    .count();
    if (sink_ != nullptr)
        *sink_ += milliseconds;
    if (report_ != nullptr)
        report_->add(name_, milliseconds);
}

} // namespace wevision
