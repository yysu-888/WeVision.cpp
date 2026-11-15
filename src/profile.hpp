#ifndef WEVISION_PROFILE_HPP
#define WEVISION_PROFILE_HPP

#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace wevision {

// All diagnostic environment toggles are parsed here so that every knob is
// documented in one place. Call sites never read the environment directly.
struct DebugOptions {
    // WEVISION_OP_PROFILE: per-op timing inside GraphSession::compute_impl.
    // Enabled unless unset, empty, or "0".
    bool operation_profile = false;
    // WEVISION_DISABLE_STAGE_PROFILE: run stage graphs without split timing.
    bool stage_profile_disabled = false;
    // WEVISION_MASK_PROFILE_DETAILS: print per-detail mask stage timings.
    bool mask_profile_details = false;
    // WEVISION_MASK_S1_F32_DIRECT: force the exact F32 direct convolution
    // instead of the faster lowered path in mask stage-1 convolutions.
    bool mask_s1_f32_direct = false;

    static DebugOptions from_environment();
};

// Accumulates named wall-clock timings in milliseconds. Timings under the
// same name add up, so repeated scopes merge automatically.
class TimingReport {
  public:
    void add(std::string_view name, double milliseconds);
    // Adds every entry of another report into this one. Use it to fold
    // per-batch reports into a total instead of hand-written field adds.
    void merge(const TimingReport &other);
    // Returns the accumulated milliseconds for a name; 0.0 if absent.
    double get(std::string_view name) const;
    // One "prefix name=ms" line per entry, sorted by name.
    void print(FILE *out, std::string_view prefix) const;
    void clear();
    bool empty() const;

  private:
    std::unordered_map<std::string, double> entries_;
};

// RAII wall-clock timer. Two sinks are supported:
//   double milliseconds = 0.0;
//   { ScopedTimer timer(&milliseconds); do_work(); }  // *sink += elapsed
//   TimingReport report;
//   { ScopedTimer timer("stage.name", report); do_work(); }
class ScopedTimer {
  public:
    explicit ScopedTimer(double *sink);
    ScopedTimer(std::string name, TimingReport &report);
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;
    ~ScopedTimer();

  private:
    double *sink_ = nullptr;
    std::string name_;
    TimingReport *report_ = nullptr;
    std::chrono::steady_clock::time_point started_;
};

} // namespace wevision

#endif
