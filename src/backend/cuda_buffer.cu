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
    std::cout << size_ << " " << values.size() << std::endl;
    //cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
}

// Vector bools perform bit packing so we must specialize
// template <>
// void CudaBuffer<bool>::write_flat(const std::vector<bool>& values) {
//     assert(sizeof(bool) == sizeof(uint8_t));
//     std::cout << size_ << " " << values.size() << std::endl;
//     assert(size_ == values.size());
//     std::vector<uint8_t> raw(values.size());
//     for (size_t i = 0; i < values.size(); ++i)
//         raw[i] = static_cast<uint8_t>(values[i]);
//     //cudaMemcpy(data_, raw.data(), size_ * sizeof(bool), cudaMemcpyHostToDevice);
// }

template <typename T>
std::vector<T> CudaBuffer<T>::read_flat() const {
    std::vector<T> res(size_);
    //cudaMemcpy(res.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    return res;
}

// Vector bools perform bit packing so we must specialize
template <>
std::vector<bool> CudaBuffer<bool>::read_flat() const {
    std::vector<uint8_t> raw(size_);
    //cudaMemcpy(raw.data(), data_, size_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
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

    dst[idx] = src[flat_to_data_idx(idx, ndim, shape, strides, offset)];
}

template <typename T>
static std::vector<T> read_strided_helper(const Shape& shape, const Strides& strides, size_t offset, T* data) {
    size_t numel = shape.numel();
    std::vector<T> res(numel);

    T* strided;
    cudaMalloc(&strided, numel * sizeof(T));
    
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
    //cudaMemcpy(res.data(), strided, numel * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(strided);
    return res;
}

// template <typename T>
// std::vector<T> CudaBuffer<T>::read_strided(const Shape& shape, const Strides& strides, size_t offset) const {
//     std::vector<T> res(shape.numel());
//     return res;
//     //return read_strided_helper(shape, strides, offset, data_);
// }

// Vector bools perform bit packing so we must specialize
// template <>
// std::vector<bool> CudaBuffer<bool>::read_strided(const Shape& shape, const Strides& strides, size_t offset) const {
//     //std::vector<uint8_t> raw(shape.numel());//read_strided_helper<uint8_t>(shape, strides, offset, reinterpret_cast<uint8_t*>(data_));
//     std::vector<bool> res(shape.numel());
//     //for (size_t i = 0; i < raw.size(); ++i)
//     //    res[i] = static_cast<bool>(raw[i]);
//     std::cout << "a" << res.size() << std::endl;
//     return res;
// }

template <typename T>
T CudaBuffer<T>::read_at(size_t offset) const {
    T val;
    //cudaMemcpy(&val, &data_[offset], sizeof(T), cudaMemcpyDeviceToHost);
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

    static constexpr auto table = cpu_utils::make_kernel_table<BinOp>(lambda);

    table[static_cast<size_t>(op)](
        shape.numel(), shape.size(), shape.array(),
        data_, rStrides.array(), rOffset,
        aData, aStrides.array(), aOffset,
        bData, bStrides.array(), bOffset
    );
}

template <typename T, typename U, typename V, size_t Op>
__global__ void apply_binary_kernel(
    size_t numel, size_t ndim, const ShapeArray shape,
    T* rData, const StridesArray rStrides, const size_t rOffset,
    U* aData, const StridesArray aStrides, const size_t aOffset,
    V b)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t rIdx = flat_to_data_idx(idx, ndim, shape, rStrides, rOffset);
    size_t aIdx = flat_to_data_idx(idx, ndim, shape, aStrides, aOffset);

    constexpr auto fn = cpu_utils::binop_table<T, U, V>[Op];
    rData[rIdx] = fn(aData[aIdx], b);
}

template <typename T>
template <typename U, typename V>
void CudaBuffer<T>::apply_binary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
    V b, BinOp op)
{
    assert(a->backend_type() == BackendType::Cuda);

    U* aData = static_cast<CudaBuffer<U>*>(a)->data_;

    using Kernel = void(*)(size_t, size_t, const ShapeArray,
                          T*, const StridesArray, const size_t,
                          U*, const StridesArray, const size_t,
                          V);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](size_t numel, size_t ndim, const ShapeArray shape,
                  T* rData, const StridesArray rStrides, const size_t rOffset,
                  U* aData, const StridesArray aStrides, const size_t aOffset,
                  V b) {
            int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            apply_binary_kernel<T, U, V, Op><<<blocks, THREADS_PER_BLOCK>>>(
                numel, ndim, shape,
                rData, rStrides, rOffset,
                aData, aStrides, aOffset,
                b
            );
        };
    };

    static constexpr auto table = cpu_utils::make_kernel_table<BinOp>(lambda);

    table[static_cast<size_t>(op)](
        shape.numel(), shape.size(), shape.array(),
        data_, rStrides.array(), rOffset,
        aData, aStrides.array(), aOffset,
        b
    );
}

template <typename T, typename U, size_t Op>
__global__ void apply_unary_kernel(
    size_t numel, size_t ndim, const ShapeArray shape,
    T* rData, const StridesArray rStrides, const size_t rOffset,
    U* aData, const StridesArray aStrides, const size_t aOffset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t rIdx = flat_to_data_idx(idx, ndim, shape, rStrides, rOffset);
    size_t aIdx = flat_to_data_idx(idx, ndim, shape, aStrides, aOffset);

    constexpr auto fn = cpu_utils::unop_table<T, U>[Op];
    rData[rIdx] = fn(aData[aIdx]);
}

template <typename T>
template <typename U>
void CudaBuffer<T>::apply_unary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
    UnOp op)
{
    assert(other->backend_type() == BackendType::Cuda);

    U* otherData = static_cast<CudaBuffer<U>*>(other)->data_;

    using Kernel = void(*)(size_t, size_t, const ShapeArray,
                          T*, const StridesArray, const size_t,
                          U*, const StridesArray, const size_t);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](size_t numel, size_t ndim, const ShapeArray shape,
                  T* rData, const StridesArray rStrides, const size_t rOffset,
                  U* otherData, const StridesArray otherStrides, const size_t otherOffset) {
            int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            apply_unary_kernel<T, U, Op><<<blocks, THREADS_PER_BLOCK>>>(
                numel, ndim, shape,
                rData, rStrides, rOffset,
                otherData, otherStrides, otherOffset
            );
        };
    };

    static constexpr auto table = cpu_utils::make_kernel_table<UnOp>(lambda);

    table[static_cast<size_t>(op)](
        shape.numel(), shape.size(), shape.array(),
        data_, rStrides.array(), rOffset,
        otherData, otherStrides.array(), otherOffset
    );
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

#include "cuda_buffer_inst.inc"

}
