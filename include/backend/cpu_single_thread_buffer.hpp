#ifndef BACKEND_CPU_SINGLE_THREAD_BUFFER_H
#define BACKEND_CPU_SINGLE_THREAD_BUFFER_H

#include "backend/base.hpp"
#include "backend/cpu_utils.hpp"

namespace backend {

using linalg::Shape;
using linalg::Strides;

template <typename T>
class CpuSingleThreadBuffer final : public DeviceBuffer<T> {
public:
    static CpuSingleThreadBuffer* create(size_t size) {
        size_t total = sizeof(CpuSingleThreadBuffer) + alignof(T) + sizeof(T) * size;
        void* mem = ::operator new(total);
        auto* buffer = new (mem) CpuSingleThreadBuffer(size);

        void* raw = reinterpret_cast<char*>(buffer) + sizeof(CpuSingleThreadBuffer);
        size_t space = total - sizeof(CpuSingleThreadBuffer);
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

    ~CpuSingleThreadBuffer() override = default;

    void write_flat(const std::vector<T>& values) override {
        assert(values.size() == size_);
        std::copy(values.begin(), values.end(), data_);
    }

    std::vector<T> read_flat() const override {
        std::vector<T> res(size_);
        std::copy(data_, data_ + size_, res.begin());
        return res;
    }

    std::vector<T> read_strided(const Shape& shape, const Strides& strides, size_t offset) const override {
        auto it = cpu_utils::StridedIterator(data_, shape, strides, offset);
        std::vector<T> res(shape.numel());
        for (size_t i = 0; i < res.size(); i++) {
            res[i] = *it;
            ++it;
        }
        return res;
    }

    T read_at(size_t offset) const override {
        if (offset >= size_) throw std::out_of_range("CpuSingleThreadBuffer::read_at");
        return data_[offset];
    }

    // T& at(size_t i) override {
    //     if (i >= size_) throw std::out_of_range("CpuSingleThreadBuffer::at");
    //     return data_[i];
    // }

    // const T& at(size_t i) const override {
    //     if (i >= size_) throw std::out_of_range("CpuSingleThreadBuffer::at");
    //     return data_[i];
    // }

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
                      BinOp op) {
        assert(a->backend_type() == BackendType::CpuSingleThread && 
               b->backend_type() == BackendType::CpuSingleThread);

        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(a)->data_, 
                                      shape, aStrides, aOffset);
        auto bIt = cpu_utils::StridedIterator<V>(static_cast<CpuSingleThreadBuffer<U>*>(b)->data_, 
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
        assert(a->backend_type() == BackendType::CpuSingleThread);

        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(a)->data_, 
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
        assert(other->backend_type() == BackendType::CpuSingleThread);
        auto otherIt = cpu_utils::StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(other)->data_, 
                                          shape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, shape, rStrides, rOffset);
        
        auto fn = cpu_utils::unop_table<T, U>[static_cast<size_t>(op)];

        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*otherIt);
            ++rIt; ++otherIt;
        }
    }

    void reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
                const Shape& reduceShape, T identity, BinOp op) override {
        assert(other->backend_type() == BackendType::CpuSingleThread);

        apply_binary(rShape, rStrides, rOffset,
                     this, rStrides, rOffset,
                     identity, BinOp::Pass);

        Shape shape(rShape);
        Strides strides(rStrides);
        for (auto x : reduceShape) {
            shape.push_back(x);
            strides.push_back(0);
        }

        apply_binary(shape, strides, rOffset,
                     this, strides, rOffset,
                     const_cast<DeviceBuffer<T>*>(other), otherStrides, otherOffset,
                     op);
    }

    // Reduce on last dimension
    template <typename U>
    void arg_reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                    size_t reduceDim, ArgRedOp op) {
        static_assert(std::is_same_v<T, size_t>, "arg_reduce only works with T = size_t");
        assert(other->backend_type() == BackendType::CpuSingleThread);
        
        Shape otherShape(rShape);
        otherShape.push_back(reduceDim);
        auto otherIt = cpu_utils::StridedIterator<U>(static_cast<const CpuSingleThreadBuffer<U>*>(other)->data_, 
                                          otherShape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);

        auto fn = cpu_utils::argredop_table<U>[static_cast<size_t>(op)];

        for (size_t i = 0; i < rShape.numel(); i++) {
            std::pair<U, size_t> cur {*otherIt, 0};
            ++otherIt;
            for (size_t j = 1; j < reduceDim; j++) {
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
        assert(a->backend_type() == BackendType::CpuSingleThread &&
               b->backend_type() == BackendType::CpuSingleThread);

        T* aData = static_cast<const CpuSingleThreadBuffer*>(a)->data_;
        T* bData = static_cast<const CpuSingleThreadBuffer*>(b)->data_;

        Shape outerShape(rShape);
        outerShape.pop_back(); outerShape.pop_back();
        Strides aOuterStrides(aStrides);
        aOuterStrides.pop_back(); aOuterStrides.pop_back();
        Strides bOuterStrides(bStrides);
        bOuterStrides.pop_back(); bOuterStrides.pop_back();

        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);
        auto aIt = cpu_utils::StridedIterator<T>(aData, outerShape, aOuterStrides, aOffset);
        auto bIt = cpu_utils::StridedIterator<T>(bData, outerShape, bOuterStrides, bOffset);
        
        size_t rDim0 = rShape[-2], rDim1 = rShape[-1];
        size_t aStride0 = aStrides[-2], aStride1 = aStrides[-1];
        size_t bStride0 = bStrides[-2], bStride1 = bStrides[-1];
        for (size_t outer = 0; outer < outerShape.numel(); outer++) {
            for (size_t i = 0; i < rDim0; i++) {
                for (size_t j = 0; j < rDim1; j++) {
                    *rIt = 0;
                    for (size_t k = 0; k < innerDim; k++) {
                        *rIt += aData[aIt.dataIdx + aStride0 * i + aStride1 * k] *
                                bData[bIt.dataIdx + bStride0 * k + bStride1 * j];
                    }
                    ++rIt;
                }
            }
            ++aIt;
            ++bIt;
        }
    }

    T& operator[](size_t i) {
        return data_[i];
    }

private:
    size_t size_;
    T* data_;

    template <typename>
    friend class CpuSingleThreadBuffer;

    CpuSingleThreadBuffer(size_t size) : DeviceBuffer<T>(BackendType::CpuSingleThread), size_(size) {}
};

}

#endif // BACKEND_CPU_SINGLE_THREAD_BUFFER_H