// Minimal dynamic parallel loop for offline diagnostics (never used inside timed search).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
namespace geometry_ann {
template<class Body>
void parallel_for(std::size_t count, std::size_t threads, Body body) {
    std::atomic<std::size_t> next{0};
    std::exception_ptr failure; std::mutex failure_lock;
    auto worker = [&](std::size_t thread) {
        try {
            for (std::size_t i = next++; i < count; i = next++) body(i, thread);
        } catch (...) {
            std::lock_guard guard(failure_lock); if (!failure) failure = std::current_exception();
            next = count;
        }
    };
    threads = std::max<std::size_t>(1, std::min(threads, count));
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& thread : pool) thread.join();
    if (failure) std::rethrow_exception(failure);
}
}
