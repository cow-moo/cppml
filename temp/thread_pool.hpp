#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace backend {

class ThreadPool {
public:
    ThreadPool(size_t numWorkers = std::thread::hardware_concurrency()) : inProgress_(numWorkers) {
        for (size_t i = 0; i < numWorkers; i++) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mutex_);
                        --inProgress_;
                        if (inProgress_ == 0)
                            cvDone_.notify_one();

                        cvTask_.wait(lock, [] {
                            return done_ || !tasks_.empty();
                        });

                        if (done_ && tasks.empty()) return;

                        task = tasks.front();
                        tasks.pop();
                        ++inProgress_;
                    }

                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        cvTask_.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    // Should be called by only main thread
    void enqueue(const std::function<void()>& fn) {
        std::lock_guard guard(mutex_);
        tasks_.push_back(fn);
        cvTask_.notify_one();
    }

    void wait() {
        std::unique_lock lock(mutex_);
        cvDone_.wait(lock, [] {
            return tasks.empty() && inProgress_ == 0;
        });
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks_;
    size_t inProgress_;
    bool done_;

    std::mutex mutex_;
    std::condition_variable cvTask_;
    std::condition_variable cvDone_;
};

}

#endif // THREAD_POOL_H