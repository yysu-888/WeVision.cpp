#ifndef WEVISION_PARALLEL_HPP
#define WEVISION_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace wevision {

// Leave half of the machine available to callers and other resident work.
inline size_t cpu_thread_budget() {
    static const size_t budget = [] {
        const unsigned hardware = std::thread::hardware_concurrency();
        return static_cast<size_t>(hardware > 1 ? hardware / 2 : 1);
    }();
    return budget;
}

// Runs independent work items with dynamic load balancing. Work items write to
// disjoint output storage; worker exceptions are rethrown on the calling
// thread.
template <typename Work> void parallel_for(size_t count, Work work) {
    const size_t thread_count = std::min(count, cpu_thread_budget());
    if (thread_count <= 1) {
        for (size_t index = 0; index < count; ++index)
            work(index);
        return;
    }

    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto run = [&] {
        try {
            for (size_t index = next.fetch_add(1); index < count;
                 index = next.fetch_add(1))
                work(index);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure)
                    failure = std::current_exception();
            }
            next.store(count);
        }
    };

    std::vector<std::future<void>> workers;
    workers.reserve(thread_count - 1);
    for (size_t thread = 1; thread < thread_count; ++thread)
        workers.push_back(std::async(std::launch::async, run));
    run();
    for (auto &worker : workers)
        worker.get();
    if (failure)
        std::rethrow_exception(failure);
}

} // namespace wevision

#endif
