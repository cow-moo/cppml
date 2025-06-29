#ifndef BACKEND_CUDA_BUFFER_H
#define BACKEND_CUDA_BUFFER_H

#include "backend/base.hpp"

namespace backend {

using linalg::Shape;
using linalg::Strides;

template <typename T>
class CudaBuffer final : public DeviceBuffer<T> {
public:
    CudaBuffer(size_t size);

    ~CudaBuffer() override;

    void write_flat(const std::vector<T>& values) override;

    std::vector<T> read_flat() const override;

    std::vector<T> read_strided(const Shape& shape, const Strides& strides, size_t offset) const override;

    T read_at(size_t) const override;

    //T& at(size_t i) override;

    //const T& at(size_t i) const override;

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
                      BinOp op);

    template <typename U, typename V>
    void apply_binary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                      DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
                      V b, BinOp op);

    template <typename U>
    void apply_unary(const Shape& shape, const Strides& rStrides, size_t rOffset,
                     DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                     UnOp op);

    void reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
                const Shape& reduceShape, T identity, BinOp op) override;

    // Reduce on last dimension
    template <typename U>
    void arg_reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                    size_t reduceDim, ArgRedOp op);

    void matmul(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* a, const Strides& aStrides, size_t aOffset,
                const DeviceBuffer<T>* b, const Strides& bStrides, size_t bOffset,
                size_t innerDim) override;

private:
    size_t size_;
    T* data_;

    template <typename>
    friend class CudaBuffer;
};

}

#endif // BACKEND_CUDA_BUFFER_H