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
        for (size_t i = 0; i < size_; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() {
                std::copy(it + i, 
                          it + std::min(size_, i + config::DEFAULT_CHUNK_SIZE),
                          data_ + i);
            });
        }
        pool.wait();
    }

    std::vector<T> read_flat() const override {
        std::vector<T> res(size_);
        auto it = res.begin();
        auto& pool = get_pool();
        for (size_t i = 0; i < size_; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() {
                std::copy(data_ + i, 
                          data_ + std::min(size_, i + config::DEFAULT_CHUNK_SIZE),
                          it + i);
            });
        }
        pool.wait();

        return res;
    }

    std::vector<T> read_strided(const Shape& shape, const Strides& strides, size_t offset) const override {
        auto it = cpu_utils::StridedIterator(data_, shape, strides, offset);
        std::vector<T> res(shape.numel());
        auto rIt = res.begin();

        size_t numel = shape.numel();

        auto& pool = get_pool();
        for (size_t i = 0; i < numel; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() mutable {
                for (size_t j = i; j < std::min(numel, i + config::DEFAULT_CHUNK_SIZE); j++) {
                    *rIt = *it;
                    ++rIt; ++it;
                }
            });
            rIt += config::DEFAULT_CHUNK_SIZE;
            it += config::DEFAULT_CHUNK_SIZE;
        }
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
        for (size_t i = 0; i < numel; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() mutable {
                for (size_t j = i; j < std::min(numel, i + config::DEFAULT_CHUNK_SIZE); j++) {
                    *rIt = fn(*aIt, *bIt);
                    ++rIt; ++aIt; ++bIt;
                }
            });
            rIt += config::DEFAULT_CHUNK_SIZE;
            aIt += config::DEFAULT_CHUNK_SIZE;
            bIt += config::DEFAULT_CHUNK_SIZE;
        }
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
        for (size_t i = 0; i < numel; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() mutable {
                for (size_t j = i; j < std::min(numel, i + config::DEFAULT_CHUNK_SIZE); j++) {
                    *rIt = fn(*aIt, b);
                    ++rIt; ++aIt;
                }
            });
            rIt += config::DEFAULT_CHUNK_SIZE;
            aIt += config::DEFAULT_CHUNK_SIZE;
        }
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
        for (size_t i = 0; i < numel; i += config::DEFAULT_CHUNK_SIZE) {
            pool.enqueue([=]() mutable {
                for (size_t j = i; j < std::min(numel, i + config::DEFAULT_CHUNK_SIZE); j++) {
                    *rIt = fn(*otherIt);
                    ++rIt; ++otherIt;
                }
            });
            rIt += config::DEFAULT_CHUNK_SIZE;
            otherIt += config::DEFAULT_CHUNK_SIZE;
        }
        pool.wait();
    }

    void reduce(const Shape& rShape, const Strides& rStrides, size_t rOffset,
                const DeviceBuffer<T>* other, const Strides& otherStrides, size_t otherOffset,
                const Shape& reduceShape, T identity, BinOp op) override {
        assert(other->backend_type() == BackendType::CpuMultiThread);
        //size_t cnt = 0;
        auto fn = cpu_utils::binop_table<T, T, T>[static_cast<size_t>(op)];
        auto& pool = get_pool();

        // Get full shape of other
        Shape shape(rShape);
        for (auto dim : reduceShape)
            shape.push_back(dim);

        T* otherData = static_cast<const CpuMultiThreadBuffer*>(other)->data_;

        // Intermediate sum
        auto prevIt = cpu_utils::StridedIterator<T>(otherData, shape, otherStrides, otherOffset);
        // Remaining dimension to reduce for intermediate sum
        size_t reduceDim = reduceShape.numel();

        // Do intermediate reductions until remaining dimension to reduce fits in chunk size
        while (reduceDim > config::DEFAULT_CHUNK_SIZE) {
            // Dimension to reduce to (>1 due to comparison in while loop)
            size_t newDim = ceil_div(reduceDim, config::DEFAULT_CHUNK_SIZE);

            // Intermediate sum allocation
            size_t curSize = rShape.numel() * newDim;
            T* cur = new T[curSize];
            auto curIt = cpu_utils::StridedIterator<T>(cur, {curSize}, {1}, 0);

            // thread per element in cursize
            for (size_t i = 0; i < curSize; i++) {
                *curIt = identity;
                // Ensure we don't reduce across boundaries determined by original reduceShape.numel
                size_t amt = std::min(reduceDim - (prevIt.flatIdx % reduceDim), config::DEFAULT_CHUNK_SIZE);
                //cnt++;
                //std::cout << amt << std::endl;
                pool.enqueue([=] mutable {
                    for (size_t j = 0; j < amt; j++) {
                        *curIt = fn(*curIt, *prevIt);
                        ++prevIt;
                    }
                });
                
                prevIt += amt;
                ++curIt;
            }
            pool.wait();

            if (prevIt.data != otherData)
                delete[] prevIt.data;
            prevIt = cpu_utils::StridedIterator<T>(cur, {curSize}, {1}, 0);
            reduceDim = newDim;
        }

        auto rIt = cpu_utils::StridedIterator<T>(data_, rShape, rStrides, rOffset);
        size_t rNumel = rShape.numel();
        while (rIt.flatIdx < rNumel) {
            // Number of elements in this buffer to write to
            size_t numElem = std::min(rNumel - rIt.flatIdx, config::DEFAULT_CHUNK_SIZE / reduceDim);
            //cnt++;
            //std::cout << numElem * reduceDim << std::endl;
            pool.enqueue([=] mutable { 
                for (size_t i = 0; i < numElem; i++) {
                    *rIt = identity;
                    for (size_t j = 0; j < reduceDim; j++) {
                        *rIt = fn(*rIt, *prevIt);
                        ++prevIt;
                    }
                    ++rIt;
                }
            });

            rIt += numElem;
            prevIt += numElem * reduceDim;
        }
        pool.wait();

        if (prevIt.data != otherData)
            delete[] prevIt.data;

        //std::cout << "threads dispatched: " << cnt << std::endl;
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
        while (rIt.flatIdx < rNumel) {
            size_t numElem = std::min(rNumel - rIt.flatIdx, 
                                      std::max(config::DEFAULT_CHUNK_SIZE / reduceDim, (size_t)1)); // write at least one
            pool.enqueue([=] mutable {
                for (size_t i = 0; i < numElem; i++) {
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

            rIt += numElem;
            otherIt += numElem * reduceDim;
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

        while (rIt.flatIdx < rNumel) {
            size_t numElem = std::min(rNumel - rIt.flatIdx, 
                                      std::max(config::DEFAULT_CHUNK_SIZE / innerDim, (size_t)1)); // write at least one

            pool.enqueue([=] mutable {
                while (numElem-- > 0) {
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

            rIt += numElem;
            j += numElem;
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