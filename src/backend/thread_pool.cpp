#include "backend/thread_pool.hpp"
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads) : stop_(false), tasksInProgress_(0) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    cvTask_.wait(lock, [this]() {
                        return stop_ || !tasks_.empty();
                    });

                    if (stop_ && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                    ++tasksInProgress_;
                }

                task();
                {
                    std::lock_guard lock(mutex_);
                    --tasksInProgress_;
                }
                cvDone_.notify_all();
            }
        });
    }
}

void ThreadPool::enqueue(const std::function<void()>& task) {
    {
        std::lock_guard lock(mutex_);
        tasks_.push(task);
    }
    cvTask_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock lock(mutex_);
    cvDone_.wait(lock, [this]() {
        return tasks_.empty() && tasksInProgress_ == 0;
    });
}

size_t ThreadPool::get_num_threads() {
    return workers_.size();
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    cvTask_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}