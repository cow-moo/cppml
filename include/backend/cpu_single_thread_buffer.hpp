#ifndef BACKEND_CPU_SINGLE_THREAD_BUFFER_H
#define BACKEND_CPU_SINGLE_THREAD_BUFFER_H

namespace backend {

template <typename T>
class CpuSingleThreadBuffer : public DeviceBuffer<T> {
public:
    using value_type = T;

    static CpuSingleThreadBuffer* create(size_t size) {
        void* mem = ::operator new(sizeof(CpuSingleThreadBuffer) + sizeof(T) * size);
        auto* buffer = new (mem) CpuSingleThreadBuffer(size);
        return buffer;
    }

    static void operator delete(void* p) {
        ::operator delete(p);
    }

    static void operator delete(void* p, std::size_t) {
        ::operator delete(p);
    }

    ~CpuSingleThreadBuffer() override = default;

    T& at(size_t i) override {
        if (i >= size_) throw std::out_of_range("CpuSingleThreadBuffer::at");
        return data_[i];
    }

    CpuSingleThreadBuffer& operator*() { return *this; }

    T& operator[](size_t i) {
        return data_[i];
    }

private:
    size_t size_;
    T data_[];

    CpuSingleThreadBuffer(size_t size) : DeviceBuffer<T>(Backend::CpuSingleThread), size_(size) {}
};

}

#endif // BACKEND_CPU_SINGLE_THREAD_BUFFER_H