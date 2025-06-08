#ifndef BACKEND_CPU_MULTI_THREAD_BUFFER_H
#define BACKEND_CPU_MULTI_THREAD_BUFFER_H

#include "backend/base.hpp"
#include "backend/cpu_utils.hpp"

namespace backend {

template <typename T>
class CpuMultiThreadBuffer final : public DeviceBuffer<T> {
public:
    static CpuMultiThreadBuffer* create(size_t size) {
        size_t total = sizeof(CpuMultiThreadBuffer) + alignof(T) + sizeof(T) * size;
        void* mem = ::operator new(total);
        auto* buffer = new (mem) CpuMultiThreadBuffer(size);

        void* raw = reinterpret_cast<char*>(buffer) + sizeof(CpuMultiThreadBuffer);
        size_t space = total - sizeof(CpuMultiThreadBuffer);
        void* aligned = std::align(alignof(T), sizeof(T) * size, raw, space);
        buffer->data_ = reinterpret_cast<T*>(aligned);
        assert(reinterpret_cast<uintptr_t>(buffer->data_) % alignof(T) == 0);
        return buffer;
    }

    static void operator delete(void* p) {
        ::operator delete(p);
    }

    static void operator delete(void* p, std::size_t) {
        ::operator delete(p);
    }

    ~CpuMultiThreadBuffer() override = default;

    void write_flat(const std::vector<T>& values) override {
        assert(values.size() == size_);
        std::copy(values.begin(), values.end(), data_);
    }

    T& at(size_t i) override {
        if (i >= size_) throw std::out_of_range("CpuMultiThreadBuffer::at");
        return data_[i];
    }

    const T& at(size_t i) const override {
        if (i >= size_) throw std::out_of_range("CpuMultiThreadBuffer::at");
        return data_[i];
    }

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
                      BinOp op) {
        assert(a->backend_type() == BackendType::CpuMultiThread && 
               b->backend_type() == BackendType::CpuMultiThread);

        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<U>(static_cast<CpuMultiThreadBuffer<U>*>(a)->data_, 
                                      shape, aStrides, aOffset);
        auto bIt = cpu_utils::StridedIterator<V>(static_cast<CpuMultiThreadBuffer<U>*>(b)->data_, 
                                      shape, bStrides, bOffset);
        
        auto fn = cpu_utils::binop_table<T, U, V>[static_cast<size_t>(op)];
        
        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*aIt, *bIt);
            ++rIt; ++aIt; ++bIt;
        }
    }

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      V b, BinOp op) {
        assert(a->backend_type() == BackendType::CpuMultiThread);

        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<U>(static_cast<CpuMultiThreadBuffer<U>*>(a)->data_, 
                                      shape, aStrides, aOffset);
        
        auto fn = cpu_utils::binop_table<T, U, V>[static_cast<size_t>(op)];

        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*aIt, b);
            ++rIt; ++aIt;
        }
    }

    template <typename U>
    void apply_unary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                     DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                     UnOp op) {
        assert(other->backend_type() == BackendType::CpuMultiThread);
        auto otherIt = cpu_utils::StridedIterator<U>(static_cast<CpuMultiThreadBuffer<U>*>(other)->data_, 
                                          shape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        
        auto fn = cpu_utils::unop_table<T, U>[static_cast<size_t>(op)];

        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*otherIt);
            ++rIt; ++otherIt;
        }
    }

    // Reduce on last dimension
    template <typename U>
    void arg_reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                    const DeviceBuffer<U>* other, 
                    const Shape& otherShape, const Strides& otherStrides, size_t otherOffset,
                    ArgRedOp op) {
        static_assert(std::is_same_v<T, size_t>, "arg_reduce only works with T = size_t");
        assert(other->backend_type() == BackendType::CpuMultiThread);
        auto otherIt = cpu_utils::StridedIterator<U>(static_cast<const CpuMultiThreadBuffer<U>*>(other)->data_, 
                                          otherShape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);

        auto fn = cpu_utils::argredop_table<U>[static_cast<size_t>(op)];

        size_t innerDim = otherShape.back();
        for (size_t i = 0; i < rShape.numel(); i++) {
            std::pair<U, size_t> cur {*otherIt, 0};
            ++otherIt;
            for (size_t j = 1; j < innerDim; j++) {
                fn(cur, {*otherIt, j});
                ++otherIt;
            }
            *rIt = cur.second;
            ++rIt;
        }
    }

    void matmul(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* a, const Strides& aStrides, size_t aOffset,
                const DeviceBuffer<T>* b, const Strides& bStrides, size_t bOffset,
                size_t innerDim) override {
        assert(a->backend_type() == BackendType::CpuMultiThread &&
               b->backend_type() == BackendType::CpuMultiThread);

        T* aData = static_cast<const CpuMultiThreadBuffer*>(a)->data_;
        T* bData = static_cast<const CpuMultiThreadBuffer*>(b)->data_;

        Shape outerShape(rShape);
        outerShape.pop_back(); outerShape.pop_back();
        Strides rOuterStrides(rStrides);
        rOuterStrides.pop_back(); rOuterStrides.pop_back();
        Strides aOuterStrides(aStrides);
        aOuterStrides.pop_back(); aOuterStrides.pop_back();
        Strides bOuterStrides(bStrides);
        bOuterStrides.pop_back(); bOuterStrides.pop_back();

        auto rIt = cpu_utils::StridedIterator<T>(data_, outerShape, rOuterStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<T>(aData, outerShape, aOuterStrides, aOffset);
        auto bIt = cpu_utils::StridedIterator<T>(bData, outerShape, bOuterStrides, bOffset);
        
        size_t rDim0 = rShape[-2], rDim1 = rShape[-1];
        size_t rStride0 = rStrides[-2], rStride1 = rStrides[-1];
        size_t aStride0 = aStrides[-2], aStride1 = aStrides[-1];
        size_t bStride0 = bStrides[-2], bStride1 = bStrides[-1];
        for (size_t outer = 0; outer < outerShape.numel(); outer++) {
            for (size_t i = 0; i < rDim0; i++) {
                for (size_t j = 0; j < rDim1; j++) {
                    size_t index = rOffset + rStride0 * i + rStride1 * j;
                    data_[index] = 0;
                    for (size_t k = 0; k < innerDim; k++) {
                        data_[index] += aData[aOffset + aStride0 * i + aStride1 * k] *
                                        bData[bOffset + bStride0 * k + bStride1 * j];
                    }
                }
            }
        }
    }

    CpuMultiThreadBuffer& operator*() { return *this; }

    T& operator[](size_t i) {
        return data_[i];
    }

private:
    size_t size_;
    T* data_;

    template <typename>
    friend class CpuMultiThreadBuffer;

    CpuMultiThreadBuffer(size_t size) : DeviceBuffer<T>(BackendType::CpuMultiThread), size_(size) {}
};

}

#endif // BACKEND_CPU_MULTI_THREAD_BUFFER_H