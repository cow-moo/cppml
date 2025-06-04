#ifndef BACKEND_BASE_H
#define BACKEND_BASE_H

#include <functional>

namespace backend {

enum class Backend {
    CpuSingleThread,
    CpuMultiThread,
};

template <typename T> class DeviceBuffer;

template <typename T>
class SharedBuffer {
public:
    SharedBuffer(DeviceBuffer<T>* buffer) : buffer_(buffer) {}

    SharedBuffer(const SharedBuffer& other) : buffer_(other.buffer_) {
        if (buffer_)
            buffer_->inc_ref();
    }

    SharedBuffer(SharedBuffer&& other) noexcept : buffer_(std::exchange(other.buffer_, nullptr)) {}

    ~SharedBuffer() {
        if (buffer_)
            buffer_->dec_ref();
    }

    SharedBuffer& operator=(const SharedBuffer& other) {
        if (this != &other) {
            if (buffer_)
                buffer_->dec_ref();
            buffer_ = other.buffer_;
            if (buffer_)
                buffer_->inc_ref();
        }
        return *this;
    }

    SharedBuffer& operator=(SharedBuffer&& other) noexcept {
        if (buffer_) buffer_->dec_ref();
        buffer_ = std::exchange(other.buffer_, nullptr);
        return *this;
    }

    explicit operator bool() const { return buffer_ != nullptr; }

    DeviceBuffer<T>* get() { return buffer_; }
    DeviceBuffer<T>& operator*() { return *buffer_; }
    DeviceBuffer<T>* operator->() { return buffer_; }

private:
    DeviceBuffer<T>* buffer_;
};

template <typename BufferType, typename... Args>
auto make_shared_buffer(Args&&... args) {
    using T = typename BufferType::value_type;
    return SharedBuffer<T>(BufferType::create(std::forward<Args>(args)...));
}


template <typename T>
class DeviceBuffer {
public:
    Backend backend() const { return backend_; }

    virtual T& at(size_t i) = 0;

protected:
    DeviceBuffer(Backend backend) : refs_(1), backend_(backend) {}

    virtual ~DeviceBuffer() {}

private:
    size_t refs_;
    Backend backend_;

    void inc_ref() {
        ++refs_;
    }

    void dec_ref() {
        if (--refs_ == 0)
            delete this;
    }

    friend class SharedBuffer<T>;
};

};

#endif // BACKEND_BASE_H