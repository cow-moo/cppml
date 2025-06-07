#ifndef BACKEND_BASE_H
#define BACKEND_BASE_H

#include <functional>

namespace backend {

using linalg::Shape;
using linalg::Strides;

enum class BackendType {
    CpuSingleThread,
    //CpuMultiThread,
};

enum class BinOp {
    Add,   // a + b
    Sub,   // a - b
    SubBy, // b - a (for scalar - tensor)
    Mul,   // a * b
    Div,   // a / b
    DivBy, // b / a (for scalar / tensor)
    Eq,    // a == b
    Pass,  // b (for assignment)
    Max,   // max(a, b)
    Min,   // min(a, b)
};

enum class UnOp {
    Exp,
    Log,
};

enum class ArgRedOp {
    Max,
};

template <typename T> class SharedBuffer;
template <typename T> class CpuSingleThreadBuffer;

#define BACKEND_DISPATCH(...) \
    switch (type_) { \
        case BackendType::CpuSingleThread: \
            static_cast<CpuSingleThreadBuffer<T>*>(this)->template __VA_ARGS__; \
            break; \
    }

template <typename T>
class DeviceBuffer {
public:
    BackendType backend_type() const { return type_; }

    // Writes a flattened tensor
    virtual void write_flat(const std::vector<T>& values) = 0;

    virtual T& at(size_t i) = 0;
    virtual const T& at(size_t i) const = 0;

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