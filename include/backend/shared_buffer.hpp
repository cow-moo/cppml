#ifndef BACKEND_SHARED_BUFFER_H
#define BACKEND_SHARED_BUFFER_H

#include "backend/base.hpp"

namespace backend {

template <typename T>
class SharedBuffer {
public:
    //SharedBuffer() : buffer_(nullptr) {};

    SharedBuffer(size_t size, BackendType type) {
        switch (type) {
            case BackendType::CpuSingleThread:
                buffer_ = CpuSingleThreadBuffer<T>::create(size);
                break;
            case BackendType::CpuMultiThread:
                buffer_ = CpuMultiThreadBuffer<T>::create(size);
                break;
            case BackendType::Cuda:
                buffer_ = new CudaBuffer<T>(size);
                break;
        }
    }

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
    const DeviceBuffer<T>* get() const { return buffer_; }
    DeviceBuffer<T>& operator*() { return *buffer_; }
    DeviceBuffer<T>* operator->() { return buffer_; }
    const DeviceBuffer<T>* operator->() const { return buffer_; }

private:
    DeviceBuffer<T>* buffer_;
};

}

#endif // BACKEND_SHARED_BUFFER_H