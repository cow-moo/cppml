#include "backend/cuda_buffer.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace backend {

using linalg::Shape;
using linalg::Strides;

constexpr int THREADS_PER_BLOCK = 256;

__constant__ size_t R_SHAPE[config::MAX_DIMS]; //, otherShape[config::MAX_DIMS];
__constant__ size_t R_STRIDES[config::MAX_DIMS], A_STRIDES[config::MAX_DIMS], B_STRIDES[config::MAX_DIMS];
__constant__ size_t OFFSETS[3];
__constant__ void* BUFFERS[3];
__constant__ size_t R_NDIM;
__constant__ size_t R_NUMEL;

// Input = 0 -> rShape, rStrides, offsets[0]
// Input = 1 -> rShape, aStrides, offsets[1]
// Input = 2 -> rShape, bStrides, offsets[2]
template <size_t Input>
__device__ size_t flat_to_data_idx(size_t flatIdx) {
    size_t res = OFFSETS[Input];
    for (size_t i = R_NDIM; i-- > 0;) {
        res += (flatIdx % R_SHAPE[i]) * (
            Input == 0 ? R_STRIDES[i] :
            Input == 1 ? A_STRIDES[i] :
            B_STRIDES[i]);
        flatIdx /= R_SHAPE[i];
    }
    return res;
}

template <typename T, size_t Input>
__device__ T read_at_flat(size_t flatIdx) {
    size_t dataIdx = flat_to_data_idx<Input>(flatIdx);
    return static_cast<T*>(BUFFERS[Input])[dataIdx];
}

template <typename T, size_t Input>
__device__ void write_at_flat(size_t flatIdx, T val) {
    size_t dataIdx = flat_to_data_idx<Input>(flatIdx);
    static_cast<T*>(BUFFERS[Input])[dataIdx] = val;
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
__global__ void read_strided_kernel(T* dst, const T* src) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R_NUMEL) return;

    size_t dataIdx = flat_to_data_idx<0>(idx);
    dst[idx] = src[dataIdx];
}

template <typename T>
static std::vector<T> read_strided_helper(const Shape& shape, const Strides& strides, size_t offset, T* data) {
    size_t ndim = shape.size();
    size_t numel = shape.numel();
    std::vector<T> res(numel);

    T* strided;
    cudaMalloc(&strided, numel * sizeof(T));
    
    cudaMemcpyToSymbol(R_SHAPE, shape.data(), config::MAX_DIMS * sizeof(size_t));
    cudaMemcpyToSymbol(R_STRIDES, strides.data(), config::MAX_DIMS * sizeof(size_t));
    cudaMemcpyToSymbol(OFFSETS, &offset, sizeof(size_t));
    cudaMemcpyToSymbol(R_NDIM, &ndim, sizeof(size_t));
    cudaMemcpyToSymbol(R_NUMEL, &numel, sizeof(size_t));

    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    read_strided_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(strided, data);
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

// template <typename T>
// T& CudaBuffer<T>::at(size_t i) {
//     return 0;
// }

// template <typename T>
// const T& CudaBuffer<T>::at(size_t i) const {
//     return 0;
// }

template <typename T, typename U, typename V, BinOp Op>
__global__ void apply_binary_kernel()
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    write_at_flat<T, 0>(idx, read_at_flat<U, 1>(idx) + read_at_flat<V, 2>(idx));
}

template <typename T>
template <typename U, typename V>
void CudaBuffer<T>::apply_binary(
    const Shape& shape, const Strides& rStrides, size_t rOffset,
    DeviceBuffer<U>* a, const Strides& aStrides, size_t aOffset,
    DeviceBuffer<V>* b, const Strides& bStrides, size_t bOffset,
    BinOp op) 
{
    
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
