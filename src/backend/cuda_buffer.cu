#include "backend/cuda_buffer.hpp"
#include "config.hpp"
#include "backend/cpu_utils.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace backend {

using linalg::Shape;
using linalg::Strides;

using ShapeArray = std::array<size_t, config::MAX_DIMS>;
using StridesArray = std::array<size_t, config::MAX_DIMS>;

constexpr int THREADS_PER_BLOCK = 256;

__device__ size_t flat_to_data_idx(
    size_t flatIdx, 
    const size_t ndim, 
    const ShapeArray shape,
    const StridesArray strides,
    const size_t offset) 
{
    size_t res = offset;
    for (size_t i = ndim; i-- > 0;) {
        res += (flatIdx % shape[i]) * strides[i];
        flatIdx /= shape[i];
    }
    return res;
}

template <typename T>
CudaBuffer<T>::CudaBuffer(size_t size) : DeviceBuffer<T>(BackendType::Cuda), size_(size) {
    cudaMalloc(&data_, size_ * sizeof(T));
}

template <typename T>
CudaBuffer<T>::~CudaBuffer() {
    cudaFree(data_);
}

template <typename T>
void CudaBuffer<T>::write_flat(const std::vector<T>& values) {
    cudaMemcpy(data_, values.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
}

// Vector bools perform bit packing so we must specialize
template <>
void CudaBuffer<bool>::write_flat(const std::vector<bool>& values) {
    std::vector<uint8_t> raw(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        raw[i] = static_cast<uint8_t>(values[i]);
    cudaMemcpy(data_, raw.data(), size_ * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

template <typename T>
std::vector<T> CudaBuffer<T>::read_flat() const {
    std::vector<T> res(size_);
    cudaMemcpy(res.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return res;
}

// Vector bools perform bit packing so we must specialize
template <>
std::vector<bool> CudaBuffer<bool>::read_flat() const {
    std::vector<uint8_t> raw(size_);
    cudaMemcpy(raw.data(), data_, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    std::vector<bool> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = static_cast<bool>(raw[i]);
    }
    return res;
}

template <typename T>
__global__ void read_strided_kernel(
    T* dst, 
    const T* src,
    const size_t numel,
    const size_t ndim, 
    const ShapeArray shape,
    const StridesArray strides,
    const size_t offset) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    //size_t dataIdx = flat_to_data_idx<0>(idx);
    //dst[idx] = src[dataIdx];
    dst[idx] = src[flat_to_data_idx(idx, ndim, shape, strides, offset)];
    //dst[idx] = read_at_flat<T, 0>(idx);
}

template <typename T>
static std::vector<T> read_strided_helper(const Shape& shape, const Strides& strides, size_t offset, T* data) {
    size_t numel = shape.numel();
    std::vector<T> res(numel);

    T* strided;
    cudaMalloc(&strided, numel * sizeof(T));
    
    //write_const_tensor<0>(data, shape, strides, offset);

    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    read_strided_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(
        strided,
        data,
        numel,
        shape.size(),
        shape.array(),
        strides.array(),
        offset
    );
    cudaMemcpy(res.data(), strided, numel * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(strided);
    return res;
}

template <typename T>
std::vector<T> CudaBuffer<T>::read_strided(const Shape& shape, const Strides& strides, size_t offset) const {
    return read_strided_helper(shape, strides, offset, data_);
}

// Vector bools perform bit packing so we must specialize
template <>
std::vector<bool> CudaBuffer<bool>::read_strided(const Shape& shape, const Strides& strides, size_t offset) const {
    std::vector<uint8_t> raw = read_strided_helper<uint8_t>(shape, strides, offset, reinterpret_cast<uint8_t*>(data_));
    std::vector<bool> res(size_);
    for (size_t i = 0; i < size_; ++i)
        res[i] = static_cast<bool>(raw[i]);
    return res;
}

template <typename T>
T CudaBuffer<T>::read_at(size_t offset) const {
    T val;
    cudaMemcpy(&val, &data_[offset], sizeof(T), cudaMemcpyDeviceToHost);
    return val;
}

template <typename T, typename U, typename V, size_t Op>
__global__ void apply_binary_kernel(
    size_t numel, size_t ndim, const ShapeArray shape,
    T* rData, const StridesArray rStrides, const size_t rOffset,
    U* aData, const StridesArray aStrides, const size_t aOffset,
    V* bData, const StridesArray bStrides, const size_t bOffset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t rIdx = flat_to_data_idx(idx, ndim, shape, rStrides, rOffset);
    size_t aIdx = flat_to_data_idx(idx, ndim, shape, aStrides, aOffset);
    size_t bIdx = flat_to_data_idx(idx, ndim, shape, bStrides, bOffset);

    constexpr auto fn = cpu_utils::binop_table<T, U, V>[Op];
    rData[rIdx] = fn(aData[aIdx], bData[bIdx]);
}

template <typename T>
template <typename U, typename V>
void CudaBuffer<T>::apply_binary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
    DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
    BinOp op) 
{
    assert(a->backend_type() == BackendType::Cuda &&
           b->backend_type() == BackendType::Cuda);

    U* aData = static_cast<CudaBuffer<U>*>(a)->data_;
    V* bData = static_cast<CudaBuffer<V>*>(b)->data_;

    //write_const_tensor<0>(data_, shape, rStrides, rOffset);
    //write_const_tensor<1>(aData, shape, aStrides, aOffset);
    //write_const_tensor<2>(bData, shape, bStrides, bOffset);

    using Kernel = void(*)(size_t, size_t, const ShapeArray,
                          T*, const StridesArray, const size_t,
                          U*, const StridesArray, const size_t,
                          V*, const StridesArray, const size_t);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](size_t numel, size_t ndim, const ShapeArray shape,
                  T* rData, const StridesArray rStrides, const size_t rOffset,
                  U* aData, const StridesArray aStrides, const size_t aOffset,
                  V* bData, const StridesArray bStrides, const size_t bOffset) {
            int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            apply_binary_kernel<T, U, V, Op><<<blocks, THREADS_PER_BLOCK>>>(
                numel, ndim, shape,
                rData, rStrides, rOffset,
                aData, aStrides, aOffset,
                bData, bStrides, bOffset
            );
        };
    };
    // struct KernelGen {
    //     using Kernel = void(*)(size_t, size_t, const ShapeArray,
    //                     T*, const StridesArray, const size_t,
    //                     U*, const StridesArray, const size_t,
    //                     V*, const StridesArray, const size_t);
    //     template <size_t Op>
    //     Kernel operator() {
    //         return [](size_t numel, size_t ndim, const ShapeArray shape,
    //                 T* rData, const StridesArray rStrides, const size_t rOffset,
    //                 U* aData, const StridesArray aStrides, const size_t aOffset,
    //                 V* bData, const StridesArray bStrides, const size_t bOffset) {
    //             int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    //             apply_binary_kernel<T, U, V, Op><<<blocks, THREADS_PER_BLOCK>>>(
    //                 numel, ndim, shape,
    //                 rData, rStrides, rOffset,
    //                 aData, aStrides, aOffset,
    //                 bData, bStrides, bOffset
    //             );
    //         };
    //     }
    // }

    static constexpr auto table = cpu_utils::make_kernel_table<BinOp>(lambda);

    table[static_cast<size_t>(op)](
        shape.numel(), shape.size(), shape.array(),
        data_, rStrides.array(), rOffset,
        aData, aStrides.array(), aOffset,
        bData, bStrides.array(), bOffset
    );

    // int blocks = (shape.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // apply_binary_kernel<T, U, V, BinOp::Add><<<blocks, THREADS_PER_BLOCK>>>(
    //     shape.numel(), shape.size(), shape.array(),
    //     data_, rStrides.array(), rOffset,
    //     aData, aStrides.array(), aOffset,
    //     bData, bStrides.array(), bOffset
    // );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
}

template <typename T>
template <typename U, typename V>
void CudaBuffer<T>::apply_binary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
    V b, BinOp op)
{
    
}

template <typename T>
template <typename U>
void CudaBuffer<T>::apply_unary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
    UnOp op)
{

}

template <typename T>
void CudaBuffer<T>::reduce(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
    const Shape& reduceShape, T identity, BinOp op) 
{

}

template <typename T>
template <typename U>
void CudaBuffer<T>::arg_reduce(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
    size_t reduceDim, ArgRedOp op)
{

}

template <typename T>
void CudaBuffer<T>::matmul(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<T>* a, const Strides& aStrides, size_t aOffset,
    const DeviceBuffer<T>* b, const Strides& bStrides, size_t bOffset,
    size_t innerDim)
{

}

template class CudaBuffer<float>;
template class CudaBuffer<unsigned long>;
template class CudaBuffer<bool>;
template class CudaBuffer<int>;
template class CudaBuffer<uint8_t>;

#define INSTANTIATE_APPLY_BINARY(T, U, V) \
template void CudaBuffer<T>::apply_binary<U, V>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    DeviceBuffer<V>*, const Strides&, size_t, \
    BinOp); \
template void CudaBuffer<T>::apply_binary<U, V>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    V, BinOp);

INSTANTIATE_APPLY_BINARY(float, float, float)
INSTANTIATE_APPLY_BINARY(unsigned long, unsigned long, unsigned long)
INSTANTIATE_APPLY_BINARY(int, int, int)

INSTANTIATE_APPLY_BINARY(bool, float, float)
INSTANTIATE_APPLY_BINARY(bool, unsigned long, unsigned long)
INSTANTIATE_APPLY_BINARY(bool, int, int)

#define INSTANTIATE_APPLY_UNARY(T, U) \
template void CudaBuffer<T>::apply_unary<U>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    UnOp);

INSTANTIATE_APPLY_UNARY(float, float)

INSTANTIATE_APPLY_UNARY(float, bool)
INSTANTIATE_APPLY_UNARY(unsigned long, bool)
INSTANTIATE_APPLY_UNARY(int, bool)

INSTANTIATE_APPLY_UNARY(float, unsigned long)
INSTANTIATE_APPLY_UNARY(float, int)
INSTANTIATE_APPLY_UNARY(float, uint8_t)

#define INSTANTIATE_ARG_REDUCE(U) \
template void CudaBuffer<size_t>::arg_reduce<U>( \
    const Shape&, const Strides&, size_t, \
    const DeviceBuffer<U>*, const Strides&, size_t, \
    size_t, ArgRedOp);

INSTANTIATE_ARG_REDUCE(float)

}
