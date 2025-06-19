#ifndef BACKEND_CPU_MULTI_THREAD_BUFFER_H
#define BACKEND_CPU_MULTI_THREAD_BUFFER_H

#include "backend/base.hpp"
#include "backend/cpu_utils.hpp"
#include "backend/thread_pool.hpp"

namespace backend {

static ThreadPool& get_pool() {
    static ThreadPool pool;
    return pool;
}

// As task size increases, first saturate first chunk up to min_chunk_size * 2
// Then, saturate num chunks while maintaining min chunk size
// After we hit max chunks, begin to increase chunk size
static size_t min_chunk_size = 1 << 17; //1 << 17;
static size_t max_chunks = get_pool().get_num_threads() * 2;

static inline size_t get_num_chunks(size_t taskSize) {
    // If res < max_chunks, then res * min_chunk_size <= taskSize < (res + 1) * min_chunk_size
    return std::max((size_t)1, std::min(max_chunks, taskSize / min_chunk_size));
}

inline size_t current_chunk_size = 1 << 17;

struct ChunkParamGuard {
    size_t prevMinChunkSize;
    size_t prevMaxChunks;
    ChunkParamGuard(size_t newMinChunkSize, size_t newMaxChunks) {
        prevMinChunkSize = min_chunk_size;
        prevMaxChunks = max_chunks;
        min_chunk_size = newMinChunkSize;
        max_chunks = newMaxChunks;
    }
    ~ChunkParamGuard() {
        min_chunk_size = prevMinChunkSize;
        max_chunks = prevMaxChunks;
    }
};

template <typename T>
static constexpr T ceil_div(T a, T b) {
    static_assert(std::is_integral_v<T>, "ceil_div requires integral types");
    return (a + b - 1) / b;
}

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
        auto it = values.begin();
        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(size_);
        size_t base = size_ / numChunks;
        size_t remainder = size_ % numChunks;

        size_t cur = 0;
        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=, data = data_] {
                std::copy(it + cur,
                          it + cur + chunkSize,
                          data + cur);
            });
            cur += chunkSize;
        }
        assert(cur == size_);

        pool.wait();
    }

    std::vector<T> read_flat() const override {
        std::vector<T> res(size_);
        auto it = res.begin();
        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(size_);
        size_t base = size_ / numChunks;
        size_t remainder = size_ % numChunks;
        
        size_t cur = 0;
        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=, data = data_] {
                std::copy(data + cur,
                          data + cur + chunkSize,
                          it + cur);
            });
            cur += chunkSize;
        }
        assert(cur == size_);

        pool.wait();

        return res;
    }

    std::vector<T> read_strided(const Shape& shape, const Strides& strides, size_t offset) const override {
        auto it = cpu_utils::StridedIterator(data_, shape, strides, offset);
        std::vector<T> res(shape.numel());
        auto rIt = res.begin();

        size_t numel = shape.numel();

        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(numel);
        size_t base = numel / numChunks;
        size_t remainder = numel % numChunks;

        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=] mutable {
                for (size_t j = 0; j < chunkSize; j++) {
                    *rIt = *it;
                    ++rIt; ++it;
                }
            });
            rIt += chunkSize;
            it += chunkSize;
        }
        assert(rIt == res.end());

        pool.wait();

        return res;
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
        size_t numel = shape.numel();

        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(numel);
        size_t base = numel / numChunks;
        size_t remainder = numel % numChunks;

        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=] mutable {
                for (size_t j = 0; j < chunkSize; j++) {
                    *rIt = fn(*aIt, *bIt);
                    ++rIt; ++aIt; ++bIt;
                }
            });
            rIt += chunkSize;
            aIt += chunkSize;
            bIt += chunkSize;
        }
        assert(rIt.flatIdx == numel);
        pool.wait();
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
        size_t numel = shape.numel();

        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(numel);
        size_t base = numel / numChunks;
        size_t remainder = numel % numChunks;

        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=] mutable {
                for (size_t j = 0; j < chunkSize; j++) {
                    *rIt = fn(*aIt, b);
                    ++rIt; ++aIt;
                }
            });
            rIt += chunkSize;
            aIt += chunkSize;
        }
        assert(rIt.flatIdx == numel);
        pool.wait();
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
        size_t numel = shape.numel();

        auto& pool = get_pool();

        size_t numChunks = get_num_chunks(numel);
        size_t base = numel / numChunks;
        size_t remainder = numel % numChunks;

        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=] mutable {
                for (size_t j = 0; j < chunkSize; j++) {
                    *rIt = fn(*otherIt);
                    ++rIt; ++otherIt;
                }
            });
            rIt += chunkSize;
            otherIt += chunkSize;
        }
        assert(rIt.flatIdx == numel);
        pool.wait();
    }

    void reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
                const Shape& reduceShape, T identity, BinOp op) override {
        assert(other->backend_type() == BackendType::CpuMultiThread);

        // size_t numChunks_ = 0;
        // size_t chunkSize_ = 0;
        // size_t plus_ = 0;

        auto fn = cpu_utils::binop_table<T, T, T>[static_cast<size_t>(op)];
        auto& pool = get_pool();

        Shape shape(rShape);
        for (auto dim : reduceShape)
            shape.push_back(dim);

        size_t reduceDim = reduceShape.numel();

        T* otherData = static_cast<const CpuMultiThreadBuffer*>(other)->data_;
        auto otherIt = cpu_utils::StridedIterator<T>(otherData, shape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);

        // Only need to have intermediate results if rShape small and reduceDim large
        size_t intermediateDim = std::min(max_chunks / rShape.numel(), reduceDim / min_chunk_size);
        if (intermediateDim >= 2) {
            size_t base = reduceDim / intermediateDim;
            size_t remainder = reduceDim % intermediateDim;
            std::vector<T> intermediate(rShape.numel() * intermediateDim);

            // Chunk size >= min_chunk_size
            assert (base >= min_chunk_size);
            // Total threads <= max_chunks
            assert(intermediate.size() <= max_chunks);

            // numChunks_ = intermediate.size();
            // chunkSize_ = base;
            // plus_ = remainder == 0 ? 0 : 1;

            //std::cout << "intermediate " << intermediate.size() << std::endl;

            for (size_t i = 0; i < rShape.numel(); i++) {
                for (size_t j = 0; j < intermediateDim; j++) {
                    size_t chunkSize = base + (j < remainder);
                    pool.enqueue([=, &intermediate] mutable {
                        T local = identity;
                        for (size_t k = 0; k < chunkSize; k++) {
                            local = fn(local, *otherIt);
                            ++otherIt;
                        }
                        intermediate[i * intermediateDim + j] = local;
                    });
                    otherIt += chunkSize;
                }
            }
            pool.wait();

            for (size_t i = 0; i < rShape.numel(); i++) {
                *rIt = identity;
                for (size_t j = 0; j < intermediateDim; j++) {
                    *rIt = fn(*rIt, intermediate[i * intermediateDim + j]);
                }
                ++rIt;
            }
        }
        else {
            size_t minNumOuter = ceil_div(min_chunk_size, reduceDim);
            size_t numChunks = std::max((size_t)1, std::min(max_chunks, rShape.numel() / minNumOuter));
            size_t base = rShape.numel() / numChunks;
            size_t remainder = rShape.numel() % numChunks;

            // numChunks_ = numChunks;
            // chunkSize_ = base * reduceDim;
            // plus_ = remainder == 0 ? 0 : reduceDim;

            for (size_t i = 0; i < numChunks; i++) {
                size_t chunkSize = base + (i < remainder);
                pool.enqueue([=] mutable {
                    for (size_t j = 0; j < chunkSize; j++) {
                        *rIt = identity;
                        for (size_t k = 0; k < reduceDim; k++) {
                            *rIt = fn(*rIt, *otherIt);
                            ++otherIt;
                        }
                        ++rIt;
                    }
                });
                rIt += chunkSize;
                otherIt += chunkSize * reduceDim;
            }
            pool.wait();
        }

        // std::cout << numChunks_ << " x (" << chunkSize_ << " + " << plus_ << ")" << std::endl;
    }


    // Reduce on last dimension
    // Currently, work per thread is at least reduceDim (so unbounded)
    // Could optimize like reduce to do intermediate reductions
    template <typename U>
    void arg_reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                    const DeviceBuffer<U>* other, const Strides& otherStrides, size_t otherOffset,
                    size_t reduceDim, ArgRedOp op) {
        static_assert(std::is_same_v<T, size_t>, "arg_reduce only works with T = size_t");
        assert(other->backend_type() == BackendType::CpuMultiThread);
        Shape otherShape(rShape);
        otherShape.push_back(reduceDim);
        auto otherIt = cpu_utils::StridedIterator<U>(static_cast<const CpuMultiThreadBuffer<U>*>(other)->data_, 
                                          otherShape, otherStrides, otherOffset);
        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);

        auto fn = cpu_utils::argredop_table<U>[static_cast<size_t>(op)];
        auto& pool = get_pool();

        size_t rNumel = rShape.numel();
        size_t minNumOuter = ceil_div(min_chunk_size, reduceDim);
        size_t numChunks = std::max((size_t)1, std::min(max_chunks, rNumel / minNumOuter));
        size_t base = rNumel / numChunks;
        size_t remainder = rNumel % numChunks;

        // size_t numChunks_ = numChunks;
        // size_t chunkSize_ = base * reduceDim;
        // size_t plus_ = remainder == 0 ? 0 : reduceDim;
        
        //std::cout << numChunks_ << " x (" << chunkSize_ << " + " << plus_ << ")" << std::endl;

        for (size_t i = 0; i < numChunks; i++) {
            size_t chunkSize = base + (i < remainder);
            pool.enqueue([=] mutable {
                for (size_t i = 0; i < chunkSize; i++) {
                    std::pair<U, size_t> cur {*otherIt, 0};
                    ++otherIt;
                    for (size_t j = 1; j < reduceDim; j++) {
                        fn(cur, {*otherIt, j});
                        ++otherIt;
                    }
                    *rIt = cur.second;
                    ++rIt;
                }
            });

            rIt += chunkSize;
            otherIt += chunkSize * reduceDim;
        }
        pool.wait();
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

        size_t rNumel = rShape.numel();
        size_t i = 0, j = 0;

        auto& pool = get_pool();

        size_t minNumOuter = ceil_div(min_chunk_size, innerDim);
        size_t numChunks = std::max((size_t)1, std::min(max_chunks, rNumel / minNumOuter));
        size_t base = rNumel / numChunks;
        size_t remainder = rNumel / numChunks;

        // size_t numChunks_ = numChunks;
        // size_t chunkSize_ = base * innerDim;
        // size_t plus_ = remainder == 0 ? 0 : innerDim;

        //std::cout << numChunks_ << " x (" << chunkSize_ << " + " << plus_ << ")" << std::endl;

        for (size_t c = 0; c < numChunks; c++) {
            size_t chunkSize = base + (c < remainder);
            pool.enqueue([=] mutable {
                while (chunkSize-- > 0) {
                    *rIt = 0;
                    for (size_t k = 0; k < innerDim; k++) {
                        *rIt += aData[aIt.dataIdx + aStride0 * i + aStride1 * k] *
                                bData[bIt.dataIdx + bStride1 * j + bStride0 * k];
                    }
                    ++rIt;
                    
                    j++;
                    if (j == rDim1) {
                        j = 0;
                        i++;
                        if (i == rDim0) {
                            i = 0;
                            ++aIt;
                            ++bIt;
                        }
                    }
                }
            });

            rIt += chunkSize;
            j += chunkSize;
            if (j >= rDim1) {
                i += j / rDim1;
                j %= rDim1;
                if (i >= rDim0) {
                    aIt += i / rDim0;
                    bIt += i / rDim0;
                    i %= rDim0;
                }
            }
        }
        //std::cout << cnt << std::endl;
        pool.wait();
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