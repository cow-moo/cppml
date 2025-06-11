#ifndef SCOPED_TIMER_H
#define SCOPED_TIMER_H

#include <chrono>
#include <iostream>
#include <string>

namespace timing {

class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start_).count();
        std::cout << name_ << ": " << duration << " ms" << std::endl;
    }
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

}

#endif // SCOPED_TIMER_H