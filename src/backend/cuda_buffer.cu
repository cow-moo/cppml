#include "backend/cuda_buffer.hpp"
#include "config.hpp"
#include "backend/cpu_utils.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <bit>
#include <cstddef>
#include <limits>

namespace backend {

using linalg::Shape;
using linalg::Strides;

using ShapeArray = std::array<size_t, config::MAX_DIMS>;
using StridesArray = std::array<size_t, config::MAX_DIMS>;

constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t MAX_REDUCE_DIM = 1024;

template <typename T>
static constexpr T ceil_div(T a, T b) {
    static_assert(std::is_integral_v<T>, "ceil_div requires integral types");
    return (a + b - 1) / b;
}

constexpr size_t bit_ceil(size_t x) {
    if (x <= 1) return 1;

    // Find the position of the highest bit set
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFF
    x |= x >> 32; // Needed for 64-bit size_t
#endif
    return x + 1;
}

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
    //std::cout << size_ << " " << values.size() << std::endl;
    assert(size_ == values.size());
    cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
}

// Vector bools perform bit packing so we must specialize
template <>
void CudaBuffer<bool>::write_flat(const std::vector<bool>& values) {
    assert(sizeof(bool) == sizeof(uint8_t));
    //std::cout << size_ << " " << values.size() << std::endl;
    assert(size_ == values.size());
    std::vector<uint8_t> raw(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        raw[i] = static_cast<uint8_t>(values[i]);
    cudaMemcpy(data_, raw.data(), size_ * sizeof(bool), cudaMemcpyHostToDevice);
}

template <typename T>
std::vector<T> CudaBuffer<T>::read_flat() const {
    std::vector<T> res(size_);
    cudaMemcpy(res.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
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

    dst[idx] = src[flat_to_data_idx(idx, ndim, shape, strides, offset)];
}

template <typename T>
static std::vector<T> read_strided_helper(const Shape& shape, const Strides& strides, size_t offset, T* data) {
    size_t numel = shape.numel();
    std::vector<T> res(numel);

    T* strided;
    cudaMalloc(&strided, numel * sizeof(T));
    
    int blocks = ceil_div(numel, THREADS_PER_BLOCK);
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
    std::vector<bool> res(shape.numel());
    for (size_t i = 0; i < raw.size(); ++i)
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
    using Kernel = void(*)(size_t, size_t, const ShapeArray,
                          T*, const StridesArray, const size_t,
                          U*, const StridesArray, const size_t,
                          V*, const StridesArray, const size_t);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](size_t numel, size_t ndim, const ShapeArray shape,
                  T* rData, const StridesArray rStrides, const size_t rOffset,
                  U* aData, const StridesArray aStrides, const size_t aOffset,
                  V* bData, const StridesArray bStrides, const size_t bOffset) {
            int blocks = ceil_div(numel, THREADS_PER_BLOCK);
            apply_binary_kernel<T, U, V, Op><<<blocks, THREADS_PER_BLOCK>>>(
                numel, ndim, shape,
                rData, rStrides, rOffset,
                aData, aStrides, aOffset,
                bData, bStrides, bOffset
            );
        };
    };
    static constexpr auto table = cpu_utils::make_kernel_table<BinOp>(lambda);

    assert(a->backend_type() == BackendType::Cuda &&
           b->backend_type() == BackendType::Cuda);

    U* aData = static_cast<CudaBuffer<U>*>(a)->data_;
    V* bData = static_cast<CudaBuffer<V>*>(b)->data_;

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

    assert(a->backend_type() == BackendType::Cuda);

    U* aData = static_cast<CudaBuffer<U>*>(a)->data_;

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

    assert(other->backend_type() == BackendType::Cuda);

    U* otherData = static_cast<CudaBuffer<U>*>(other)->data_;

    table[static_cast<size_t>(op)](
        shape.numel(), shape.size(), shape.array(),
        data_, rStrides.array(), rOffset,
        otherData, otherStrides.array(), otherOffset
    );
}

template <typename T, size_t Op>
__global__ void reduce_kernel_final(
    const size_t rNdim, const ShapeArray rShape, 
    T* rData, const StridesArray rStrides, const size_t rOffset,
    const size_t reduceDim, const size_t otherNdim, const ShapeArray otherShape, 
    T* otherData, const StridesArray otherStrides, const size_t otherOffset)
{
    __shared__ T sdata[MAX_REDUCE_DIM];

    assert(reduceDim <= MAX_REDUCE_DIM);
    assert(blockDim.x >= reduceDim);

    size_t tid = threadIdx.x;

    if (threadIdx.x < reduceDim) {
        size_t otherFlatIdx = blockIdx.x * reduceDim + threadIdx.x;
        size_t otherDataIdx = flat_to_data_idx(otherFlatIdx, otherNdim, otherShape, otherStrides, otherOffset);
        sdata[tid] = otherData[otherDataIdx];
    }
    __syncthreads();

    constexpr auto fn = cpu_utils::binop_table<T, T, T>[Op];
    for (size_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && tid + s < reduceDim)
            sdata[tid] = fn(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        size_t rDataIdx = flat_to_data_idx(blockIdx.x, rNdim, rShape, rStrides, rOffset);
        rData[rDataIdx] = sdata[0];
    }
}

// Owns newData and assumes flat (default strides, 0 offset)
// newShape[-1] is new reduce dim
// curData guaranteed to share a prefix with newShape[:-1]
template <typename T, size_t Op>
__global__ void reduce_kernel(
    T* newData, const size_t newReduceDim,
    const T* curData, const size_t curNdim,
    const ShapeArray curShape, const StridesArray curStrides, const size_t curOffset,
    const size_t curReduceDim)
{
    // Trying to use extern dynamic sizing leads to some symbol collision
    // from the explicit instantiation of CudaBuffer
    __shared__ T sdata[MAX_REDUCE_DIM];

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t bdim = blockDim.x;

    size_t finalIdx = bid / newReduceDim;
    size_t intermediateIdx = bid % newReduceDim;
    // We can describe any position in newData (finalIdx, intermediateIdx)
    // For each of these we want to reduce starting from curData (finalIdx, intermediateIdx * bdim)
    // Up to bdim elements or until we hit a reduction boundary
    // Each finalIdx corresponds to an index in final reduction
    assert(curReduceDim > intermediateIdx * bdim);
    size_t blockReduceDim = min(curReduceDim - intermediateIdx * bdim, bdim);
    if (tid < blockReduceDim) {
        size_t curFlatIdx = finalIdx * curReduceDim + intermediateIdx * bdim + tid;
        size_t curDataIdx = flat_to_data_idx(curFlatIdx, curNdim, curShape, curStrides, curOffset);
        sdata[tid] = curData[curDataIdx];
    }
    __syncthreads();

    constexpr auto fn = cpu_utils::binop_table<T, T, T>[Op];
    for (size_t s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockReduceDim)
            sdata[tid] = fn(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        newData[bid] = sdata[0];
    }
}

template <typename T>
void CudaBuffer<T>::reduce(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
    const Shape& reduceShape, T identity, BinOp op) 
{
    using Kernel = void(*)(T*, const size_t,
                           const T*, const size_t,
                           const ShapeArray, const StridesArray, const size_t,
                           const size_t);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](T* newData, const size_t newReduceDim,
                  const T* curData, const size_t curNdim,
                  const ShapeArray curShape, const StridesArray curStrides, const size_t curOffset,
                  const size_t curReduceDim) {
            size_t newSize = newReduceDim;
            for (size_t i = 0; i < curNdim; ++i)
                newSize *= curShape[i];
            newSize /= curReduceDim;

            size_t bdim = bit_ceil(ceil_div(curReduceDim, newReduceDim));
            assert(bdim <= MAX_REDUCE_DIM);
            // newReduceDim != 1 implies bdim == MAX_REDUCE_DIM
            assert(newReduceDim == 1 || bdim == MAX_REDUCE_DIM);

            reduce_kernel<T, Op><<<newSize, bdim>>>(
                newData, newReduceDim,
                curData, curNdim,
                curShape, curStrides, curOffset,
                curReduceDim
            );
        };
    };
    static constexpr auto table = cpu_utils::make_kernel_table<BinOp>(lambda);

    assert(other->backend_type() == BackendType::Cuda);

    T* otherData = static_cast<const CudaBuffer*>(other)->data_;
    size_t rNumel = rShape.numel();

    // Get full shape of other
    Shape curShape(rShape);
    for (auto dim : reduceShape)
        curShape.push_back(dim);

    size_t curNdim = curShape.size();
    T* curData = otherData;
    Strides curStrides = otherStrides;
    size_t curOffset = otherOffset;
    // Remaining dimension to reduce for intermediate sum
    size_t curReduceDim = reduceShape.numel();

    // Do intermediate reductions until remaining dimension to reduce fits in chunk size
    while (curReduceDim > MAX_REDUCE_DIM) {
        // Dimension to reduce to (>1 due to comparison in while loop)
        size_t newReduceDim = ceil_div(curReduceDim, MAX_REDUCE_DIM);

        // Intermediate sum allocation
        size_t newSize = rNumel * newReduceDim;
        T* newData;
        cudaMalloc(&newData, newSize * sizeof(T));

        // Dispatch kernel
        table[static_cast<size_t>(op)](
            newData, newReduceDim,
            curData, curNdim,
            curShape.array(), curStrides.array(), curOffset,
            curReduceDim
        );

        // Free curData if it was allocated
        if (curData != otherData)
            cudaFree(curData);
        
        // Update cur
        curNdim = rShape.size() + 1;
        curShape = rShape;
        curShape.push_back(newReduceDim);
        curData = newData;
        curStrides = Strides(curShape);
        curOffset = 0;
        curReduceDim = newReduceDim;
    }

    table[static_cast<size_t>(op)](
        data_, 1,
        curData, curNdim,
        curShape.array(), curStrides.array(), curOffset,
        curReduceDim
    );

    if (curData != otherData)
        cudaFree(curData);
}

// Owns newData and assumes flat (default strides, 0 offset)
// newShape[-1] is new reduce dim
// curData guaranteed to share a prefix with newShape[:-1]
// if curIdxs == nullptr then assume default strides
// if newData == nullptr then don't write
template <typename T, size_t Op>
__global__ void arg_reduce_kernel(
    T* newData, size_t* newIdxs, const size_t newReduceDim,
    const T* curData, const size_t* curIdxs, const size_t curNdim,
    const ShapeArray curShape, const StridesArray curStrides, const size_t curOffset,
    const size_t curReduceDim)
{
    // Trying to use extern dynamic sizing leads to some symbol collision
    // from the explicit instantiation of CudaBuffer
    __shared__ T sdata[MAX_REDUCE_DIM];
    __shared__ size_t sidxs[MAX_REDUCE_DIM];

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t bdim = blockDim.x;

    size_t finalIdx = bid / newReduceDim;
    size_t intermediateIdx = bid % newReduceDim;
    // We can describe any position in newData (finalIdx, intermediateIdx)
    // For each of these we want to reduce starting from curData (finalIdx, intermediateIdx * bdim)
    // Up to bdim elements or until we hit a reduction boundary
    // Each finalIdx corresponds to an index in final reduction
    assert(curReduceDim > intermediateIdx * bdim);
    size_t blockReduceDim = min(curReduceDim - intermediateIdx * bdim, bdim);
    if (tid < blockReduceDim) {
        size_t curFlatIdx = finalIdx * curReduceDim + intermediateIdx * bdim + tid;
        size_t curDataIdx = flat_to_data_idx(curFlatIdx, curNdim, curShape, curStrides, curOffset);
        sdata[tid] = curData[curDataIdx];
        sidxs[tid] = curIdxs == nullptr ? intermediateIdx * bdim + tid : curIdxs[curDataIdx];
    }
    __syncthreads();

    for (size_t s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockReduceDim) {
            if constexpr (Op == static_cast<size_t>(ArgRedOp::Max)) {
                if (sdata[tid] < sdata[tid + s]) {
                    sdata[tid] = sdata[tid + s];
                    sidxs[tid] = sidxs[tid + s];
                }
            }
            else {
                if (sdata[tid] > sdata[tid + s]) {
                    sdata[tid] = sdata[tid + s];
                    sidxs[tid] = sidxs[tid + s];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (newData != nullptr)
            newData[bid] = sdata[0];
        newIdxs[bid] = sidxs[0];
    }
}

template <typename T>
template <typename U>
void CudaBuffer<T>::arg_reduce(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
    size_t reduceDim, ArgRedOp op)
{
    static_assert(std::is_same_v<T, size_t>, "arg_reduce only works with T = size_t");

    using Kernel = void(*)(U*, size_t*, const size_t,
                           const U*, const size_t*, const size_t,
                           const ShapeArray, const StridesArray, const size_t,
                           const size_t);
    static constexpr auto lambda = []<size_t Op>() -> Kernel {
        return [](U* newData, size_t* newIdxs, const size_t newReduceDim,
                  const U* curData, const size_t* curIdxs, const size_t curNdim,
                  const ShapeArray curShape, const StridesArray curStrides, const size_t curOffset,
                  const size_t curReduceDim) {
            size_t newSize = newReduceDim;
            for (size_t i = 0; i < curNdim; ++i)
                newSize *= curShape[i];
            newSize /= curReduceDim;

            size_t bdim = bit_ceil(ceil_div(curReduceDim, newReduceDim));
            assert(bdim <= MAX_REDUCE_DIM);
            // newReduceDim != 1 implies bdim == MAX_REDUCE_DIM
            assert(newReduceDim == 1 || bdim == MAX_REDUCE_DIM);

            arg_reduce_kernel<U, Op><<<newSize, bdim>>>(
                newData, newIdxs, newReduceDim,
                curData, curIdxs, curNdim,
                curShape, curStrides, curOffset,
                curReduceDim
            );
        };
    };
    static constexpr auto table = cpu_utils::make_kernel_table<ArgRedOp>(lambda);

    assert(other->backend_type() == BackendType::Cuda);

    U* otherData = static_cast<const CudaBuffer<U>*>(other)->data_;
    size_t rNumel = rShape.numel();

    // Get full shape of other
    Shape curShape(rShape);
    curShape.push_back(reduceDim);

    size_t curNdim = curShape.size();
    U* curData = otherData;
    size_t* curIdxs = nullptr;
    Strides curStrides = otherStrides;
    size_t curOffset = otherOffset;
    // Remaining dimension to reduce for intermediate sum
    size_t curReduceDim = reduceDim;

    // Do intermediate reductions until remaining dimension to reduce fits in chunk size
    while (curReduceDim > MAX_REDUCE_DIM) {
        // Dimension to reduce to (>1 due to comparison in while loop)
        size_t newReduceDim = ceil_div(curReduceDim, MAX_REDUCE_DIM);

        // Intermediate sum allocation
        size_t newSize = rNumel * newReduceDim;
        U* newData;
        size_t* newIdxs;
        cudaMalloc(&newData, newSize * sizeof(U));
        cudaMalloc(&newIdxs, newSize * sizeof(size_t));

        // Dispatch kernel
        table[static_cast<size_t>(op)](
            newData, newIdxs, newReduceDim,
            curData, curIdxs, curNdim,
            curShape.array(), curStrides.array(), curOffset,
            curReduceDim
        );

        // Free curData if it was allocated
        if (curData != otherData) {
            cudaFree(curData);
            cudaFree(curIdxs);
        }
        
        // Update cur
        curNdim = rShape.size() + 1;
        curShape = rShape;
        curShape.push_back(newReduceDim);
        curData = newData;
        curIdxs = newIdxs;
        curStrides = Strides(curShape);
        curOffset = 0;
        curReduceDim = newReduceDim;
    }

    table[static_cast<size_t>(op)](
        nullptr, data_, 1,
        curData, curIdxs, curNdim,
        curShape.array(), curStrides.array(), curOffset,
        curReduceDim
    );

    if (curData != otherData) {
        cudaFree(curData);
        cudaFree(curIdxs);
    }
}

constexpr size_t BM = 64;
constexpr size_t BN = 64;
constexpr size_t BK = 32;

constexpr size_t TM = 4;
constexpr size_t TN = 4;

constexpr size_t NTHREADS = BM * BN / TM / TN;

static_assert(NTHREADS % BK == 0);
constexpr size_t A_STRIDE_INNER = NTHREADS / BK;
static_assert(NTHREADS % BN == 0);
constexpr size_t B_STRIDE_INNER = NTHREADS / BN;

// BS * M * K
// BS * K * N
// BS * M * N
// BS * ceil_div(M, TM) * ceil_div(N, TN)
// BS * ceil_div(M, BM) * ceil_div(N, BN) blocks
// 1 * (BM / TM) * (BN / TN) threads per block
// z, x, y
template <typename T>
__global__ void matmul_kernel(
    const size_t ndim, const ShapeArray batchShape, 
    const size_t m, const size_t n, const size_t k,
    T* rData, const StridesArray rStrides, const size_t rOffset,
    const T* aData, const StridesArray aStrides, const size_t aOffset,
    const T* bData, const StridesArray bStrides, const size_t bOffset)
{
    __shared__ T sa[BM * BK];
    __shared__ T sb[BK * BN];

    size_t batchIdx = blockIdx.z;
    size_t rBatchOffset = flat_to_data_idx(batchIdx, ndim, batchShape, rStrides, rOffset);
    size_t aBatchOffset = flat_to_data_idx(batchIdx, ndim, batchShape, aStrides, aOffset);
    size_t bBatchOffset = flat_to_data_idx(batchIdx, ndim, batchShape, bStrides, bOffset);

    size_t tid = threadIdx.x + threadIdx.y * blockDim.x;

    size_t aIdxM = blockIdx.x * BM + tid / BK;
    size_t aIdxK = tid % BK;

    size_t bIdxK = tid / BN;
    size_t bIdxN = blockIdx.y * BN + tid % BN;

    size_t rIdxM = threadIdx.x * TM + blockIdx.x * BM;
    size_t rIdxN = threadIdx.y * TN + blockIdx.y * BN;

    size_t tm = min(TM, m > rIdxM ? m - rIdxM : 0);
    size_t tn = min(TN, n > rIdxN ? n - rIdxN : 0);

    T res[TM * TN] = {0};
    T aReg[TM], bReg[TN];

    // Outer loop
    for (size_t kIdx = 0; kIdx < k; kIdx += BK) {
        // Load SMEM
        if (aIdxK < k) {
            for (size_t i = 0; i < BM * BK / NTHREADS; ++i) {
                if (aIdxM + i * A_STRIDE_INNER < m) {
                    sa[i * NTHREADS + tid] = 
                        aData[aBatchOffset + 
                              (aIdxM + i * A_STRIDE_INNER) * aStrides[ndim] +
                              aIdxK * aStrides[ndim + 1]];
                }
            }
        }

        if (bIdxN < n) {
            for (size_t i = 0; i < BK * BN / NTHREADS; ++i) {
                if (bIdxK + i * B_STRIDE_INNER < k) {
                    sb[i * NTHREADS + tid] =
                        bData[bBatchOffset +
                              (bIdxK + i * B_STRIDE_INNER) * bStrides[ndim] +
                              bIdxN * bStrides[ndim + 1]];
                }
            }
        }

        __syncthreads();

        aIdxK += BK;
        bIdxK += BK;

        // Inner loop
        for (size_t i = 0; i < min(BK, k - kIdx); ++i) {
            // Load regs
            for (size_t j = 0; j < tm; ++j) {
                aReg[j] = sa[i + (j + threadIdx.x * TM) * BK];
            }

            for (size_t k = 0; k < tn; ++k) {
                bReg[k] = sb[k + threadIdx.y * TN + i * BN];
            }

            // Outer product
            for (size_t j = 0; j < tm; ++j) {
                for (size_t k = 0; k < tn; ++k) {
                    res[k + j * TN] += aReg[j] * bReg[k];
                }
            }
        }

        __syncthreads();
    }

    size_t rThreadOffset = rBatchOffset + rIdxM * rStrides[ndim] + rIdxN * rStrides[ndim + 1];

    // Write results
    for (size_t j = 0; j < tm; ++j) {
        for (size_t k = 0; k < tn; ++k) {
            rData[rThreadOffset + j * rStrides[ndim] + k * rStrides[ndim + 1]] = res[k + j * TN];
        }
    }
}

template <typename T>
void CudaBuffer<T>::matmul(
    const Shape& rShape, const Strides& rStrides, size_t rOffset,
    const DeviceBuffer<T>* a, const Strides& aStrides, size_t aOffset,
    const DeviceBuffer<T>* b, const Strides& bStrides, size_t bOffset,
    size_t innerDim)
{
    Shape batchShape = rShape;
    batchShape.pop_back(); batchShape.pop_back();
    size_t m = rShape[-2];
    size_t n = rShape[-1];

    assert(a->backend_type() == BackendType::Cuda &&
           b->backend_type() == BackendType::Cuda);

    const T* aData = static_cast<const CudaBuffer*>(a)->data_;
    const T* bData = static_cast<const CudaBuffer*>(b)->data_;

    constexpr dim3 blockDim(BM / TM, BN / TN, 1);
    dim3 gridDim(ceil_div(m, BM), ceil_div(n, BN), batchShape.numel());
    matmul_kernel<T><<<gridDim, blockDim>>>(
        batchShape.size(), batchShape.array(),
        m, n, innerDim,
        data_, rStrides.array(), rOffset,
        aData, aStrides.array(), aOffset,
        bData, bStrides.array(), bOffset
    );
}

#include "cuda_buffer_inst.inc"

}

    // size_t aBlockOffset = blockDim.x * BM * aStrides[ndim];
    // size_t bBlockOffset = blockDim.y * BN * bStrides[ndim + 1];

    // size_t aThreadOffset = tid / BK * aStrides[ndim] + tid % BK * aStrides[ndim + 1];
    // size_t bThreadOffset = tid / BN * bStrides[ndim] + tid % BN * bStrides[ndim + 1];

    // size_t aStrideOuter = BK * aStrides[ndim + 1];
    // size_t aStrideInner = NTHREADS / BK * aStrides[ndim];
    // static_assert(NTHREADS % BK == 0);

    // size_t bStrideOuter = BK * bStrides[ndim];
    // size_t bStrideInner = NTHREADS / BN * bStrides[ndim];
    // static_assert(NTHREADS % BN == 0);

    // size_t aIdx = aBatchOffset + aBlockOffset + aThreadOffset;
    // size_t bIdx = bBatchOffset + bBlockOffset + bThreadOffset;