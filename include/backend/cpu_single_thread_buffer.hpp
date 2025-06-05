#ifndef BACKEND_CPU_SINGLE_THREAD_BUFFER_H
#define BACKEND_CPU_SINGLE_THREAD_BUFFER_H

#include "backend/base.hpp"

namespace backend {

using linalg::Shape;
using linalg::Strides;

template <typename T, typename U, typename V>
using BinOpFn = T(*)(U, V);

template <typename T, typename U, typename V>
static constexpr BinOpFn<T, U, V> binop_table[] = {
    [](U x, V y) -> T { return static_cast<T>(x + y); },     // BinOp::Add
    [](U x, V y) -> T { return static_cast<T>(x - y); },     // BinOp::Sub
    [](U x, V y) -> T { return static_cast<T>(y - x); },     // BinOp::SubBy
    [](U x, V y) -> T { return static_cast<T>(x * y); },     // BinOp::Mul
    [](U x, V y) -> T { return static_cast<T>(x / y); },     // BinOp::Div
    [](U x, V y) -> T { return static_cast<T>(y / x); },     // BinOp::DivBy
    [](U x, V y) -> T { return static_cast<T>(x == y); },    // BinOp::Eq
    [](U, V y)   -> T { return static_cast<T>(y); },         // BinOp::Pass
};

template <typename T, typename U>
using UnOpFn = T(*)(U);

template <typename T, typename U>
static constexpr UnOpFn<T, U> unop_table[] = {
    [](U x) { return std::exp(x); },
    [](U x) { return std::log(x); },
};

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

    T& at(size_t i) override {
        if (i >= size_) throw std::out_of_range("CpuSingleThreadBuffer::at");
        return data_[i];
    }

    const T& at(size_t i) const override {
        if (i >= size_) throw std::out_of_range("CpuSingleThreadBuffer::at");
        return data_[i];
    }

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
                      BinOp op) {
        assert(a->backend_type() == BackendType::CpuSingleThread && 
               b->backend_type() == BackendType::CpuSingleThread);
        auto aIt = StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(a)->data_, 
                                      shape, aStrides, aOffset);
        auto bIt = StridedIterator<V>(static_cast<CpuSingleThreadBuffer<U>*>(b)->data_, 
                                      shape, bStrides, bOffset);
        auto rIt = StridedIterator<T>(data_, shape, rStrides, rOffset);
        
        auto fn = binop_table<T, U, V>[static_cast<size_t>(op)];
        
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
        auto aIt = StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(a)->data_, 
                                      shape, aStrides, aOffset);
        auto rIt = StridedIterator<T>(data_, shape, rStrides, rOffset);
        
        auto fn = binop_table<T, U, V>[static_cast<size_t>(op)];

        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*aIt, b);
            ++rIt; ++aIt;
        }
    }

    template <typename U, typename V>
    void apply_unary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                     DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                     UnOp op) {
        assert(other->backend_type() == BackendType::CpuSingleThread);
        auto otherIt = StridedIterator<U>(static_cast<CpuSingleThreadBuffer<U>*>(other)->data_, 
                                          shape, otherStrides, otherOffset);
        auto rIt = StridedIterator<T>(data_, shape, rStrides, rOffset);
        
        auto fn = unop_table<T, U>[static_cast<size_t>(op)];

        for (size_t i = 0; i < shape.numel(); i++) {
            *rIt = fn(*otherIt);
            ++rIt; ++otherIt;
        }
    }

    CpuSingleThreadBuffer& operator*() { return *this; }

    T& operator[](size_t i) {
        return data_[i];
    }

private:
    size_t size_;
    T* data_;

    template <typename>
    friend class CpuSingleThreadBuffer;

    CpuSingleThreadBuffer(size_t size) : DeviceBuffer<T>(BackendType::CpuSingleThread), size_(size) {}

    template <typename U>
    struct StridedIterator {
        U* data;
        Shape shape;
        Strides strides;
        std::array<size_t, MAX_SBO_DIMS> idxs;
        size_t flatIdx;

        StridedIterator(U* data, const Shape& shape, const Strides& strides, size_t offset) 
            : data(data), shape(shape), strides(strides), idxs{}, flatIdx(offset) {}

        U& operator*() {
            return data[flatIdx];
        }

        StridedIterator& operator++() {
            for (int i = shape.size() - 1; i >= 0; i--) {
                flatIdx += strides[i];
                if (++idxs[i] == shape[i]) {
                    idxs[i] = 0;
                    flatIdx -= strides[i] * shape[i];
                }
                else break;
            }
            return *this;
        }

        StridedIterator& operator+=(size_t n) {
            idxs[shape.size() - 1] += n;
            flatIdx += n * strides[shape.size() - 1];

            for (int i = shape.size() - 1; i >= 1; i--) {
                if (idxs[i] >= shape[i]) {
                    size_t num = idxs[i] / shape[i];
                    idxs[i] -= num * shape[i];
                    flatIdx -= num * shape[i] * strides[i];
                    idxs[i - 1] += num;
                    flatIdx += num * strides[i - 1];
                }
                else break;
            }

            return *this;
        }
        
        StridedIterator operator+(size_t n) {
            StridedIterator res(*this);
            res += n;
            return res;
        }

        bool operator==(const StridedIterator& other) const {
            return other.data == data && other.flatIdx == flatIdx;
        }
    };
};

}

#endif // BACKEND_CPU_SINGLE_THREAD_BUFFER_H