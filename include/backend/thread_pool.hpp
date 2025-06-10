#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();

    void enqueue(const std::function<void()>& task);
    void wait(); // Wait until all tasks finish

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable cvTask_;
    std::condition_variable cvDone_;
    bool stop_;
    size_t tasksInProgress_;
};

#endif // THREAD_POOL_H