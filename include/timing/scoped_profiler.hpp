#ifndef SCOPED_PROFILER_H
#define SCOPED_PROFILER_H

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace timing {

class Profiler {
public:
    static void add_time(const std::string& name, double ms) {
        data_[name].first += ms;
        data_[name].second++;
    }

    static void report(bool avg = false) {
        std::cout << "\n=== Profiling Results ===\n";
        for (auto& [name, total] : data_) {
            std::cout << name << ": " << (avg ? total.first / total.second : total.first) << " ms" << std::endl;
        }
    }

    static void reset() {
        data_.clear();
    }

private:
    static inline std::unordered_map<std::string, std::pair<double, size_t>> data_;
};

class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedProfiler() {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start_).count();
        Profiler::add_time(name_, duration);
    }
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

}

#endif // SCOPED_PROFILER_H