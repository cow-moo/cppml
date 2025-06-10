#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <functional>
#include <queue>

namespace backend {

class ThreadPool {
public:
    ThreadPool(size_t numWorkers = std::thread::hardware_concurrency());

    // Should be called by only main thread
    void enqueue(std::function<void()> fn);



    // void enqueue(std::function<void()> fn) {
    //     tasks_.emplace(std::move(fn));
    // }
    // ThreadPool(size_t numWorkers = std::thread::hardware_concurrency()) {
    //     for (size_t i = 0; i < numWorkers; i++) {

    //     }
    // }

private:
    std::queue<std::function<void()>> tasks_;
};

}

#endif // THREAD_POOL_H