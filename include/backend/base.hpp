#ifndef BACKEND_BASE_H
#define BACKEND_BASE_H

#include <functional>
#include "backend/backend_type.hpp"
#include "linalg/shape.hpp"

namespace backend {

using linalg::Shape;
using linalg::Strides;

enum class BinOp {
    Add,   // a + b
    Sub,   // a - b
    SubBy, // b - a (for scalar - tensor)
    Mul,   // a * b
    Div,   // a / b
    DivBy, // b / a (for scalar / tensor)
    Eq,    // a == b
    Lt,    // a < b
    Lte,   // a <= b
    Gt,    // a > b (for tensor > scalar)
    Gte,   // a >= b ( for tensor >= scalar)
    Pass,  // b (for assignment)
    Max,   // max(a, b)
    Min,   // min(a, b)
};

enum class UnOp {
    Exp,
    Log,
    Neg,  // -x
    Pass, // for astype
};

enum class ArgRedOp {
    Max,
    Min,
};

template <typename T> class SharedBuffer;
template <typename T> class CpuSingleThreadBuffer;
template <typename T> class CpuMultiThreadBuffer;
template <typename T> class CudaBuffer;

#define BACKEND_DISPATCH(...) \
    switch (type_) { \
        case BackendType::CpuSingleThread: \
            static_cast<CpuSingleThreadBuffer<T>*>(this)->template __VA_ARGS__; \
            break; \
        case BackendType::CpuMultiThread: \
            static_cast<CpuMultiThreadBuffer<T>*>(this)->template __VA_ARGS__; \
            break; \
    }
    //     case BackendType::Cuda: \
    //         static_cast<CudaBuffer<T>*>(this)->template __VA_ARGS__; \
    //         break; \
    // }

template <typename T>
class DeviceBuffer {
public:
    BackendType backend_type() const { return type_; }

    // Writes a flattened tensor
    virtual void write_flat(const std::vector<T>& values) = 0;

    virtual std::vector<T> read_flat() const = 0;
    virtual std::vector<T> read_strided(const Shape&, const Strides&, size_t) const = 0;

    virtual T read_at(size_t) const = 0;
    //virtual T& at(size_t i) = 0;
    //virtual const T& at(size_t i) const = 0;

    // Fake virtual function due to template
    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset, 
                      DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
                      BinOp op) {
        BACKEND_DISPATCH(apply_binary(shape, rStrides, rOffset, a, aStrides, aOffset, b, bStrides, bOffset, op));
    }

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      V b, BinOp op) {
        BACKEND_DISPATCH(apply_binary(shape, rStrides, rOffset, a, aStrides, aOffset, b, op));
    }

    template <typename U>
    void apply_unary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                     DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                     UnOp op) {
        BACKEND_DISPATCH(apply_unary(shape, rStrides, rOffset, other, otherStrides, otherOffset, op));
    }

    // Reduce on last k dimensions (implied by reduceShape)
    // otherShape = rShape + reduceShape
    virtual void reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer* other, const Strides& otherStrides, size_t otherOffset,
                const Shape& reduceShape, T identity, BinOp op) = 0;

    // Reduce on last dimension
    template <typename U>
    void arg_reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                    size_t reduceDim, ArgRedOp op) {
        static_assert(std::is_same_v<T, size_t>, "arg_reduce only works with T = size_t");
        BACKEND_DISPATCH(arg_reduce(rShape, rStrides, rOffset, other, otherStrides, otherOffset, reduceDim, op));
    }

    virtual void matmul(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* a, const Strides& aStrides, size_t aOffset,
                const DeviceBuffer<T>* b, const Strides& bStrides, size_t bOffset,
                size_t innerDim) = 0;

protected:
    DeviceBuffer(BackendType type) : refs_(1), type_(type) {}

    virtual ~DeviceBuffer() {}

private:
    size_t refs_;
    BackendType type_;

    void inc_ref() {
        ++refs_;
    }

    void dec_ref() {
        if (--refs_ == 0)
            delete this;
    }

    friend class SharedBuffer<T>;
};

}

#endif // BACKEND_BASE_H