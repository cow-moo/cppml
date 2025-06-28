#ifndef BACKEND_CPU_UTIL_H
#define BACKEND_CPU_UTIL_H

#include <array>

namespace backend {

namespace cpu_utils {

template <typename T, typename U, typename V>
using BinOpFn = T(*)(U, V);

template <typename KernelGen, size_t... Is>
constexpr auto make_kernel_table_impl(KernelGen gen, std::index_sequence<Is...>) {
    return std::array{
        gen.template operator()<Is>()...
    };
}

template <typename E, typename KernelGen>
constexpr auto make_kernel_table(KernelGen gen) {
    constexpr size_t N = static_cast<size_t>(E::Count);
    return make_kernel_table_impl(gen, std::make_index_sequence<N>{});
}

template <typename T, typename U, typename V>
static constexpr BinOpFn<T, U, V> binop_table[] = {
    [](U x, V y) -> T { return static_cast<T>(x + y); },          // BinOp::Add
    [](U x, V y) -> T { return static_cast<T>(x - y); },          // BinOp::Sub
    [](U x, V y) -> T { return static_cast<T>(y - x); },          // BinOp::SubBy
    [](U x, V y) -> T { return static_cast<T>(x * y); },          // BinOp::Mul
    [](U x, V y) -> T { return static_cast<T>(x / y); },          // BinOp::Div
    [](U x, V y) -> T { return static_cast<T>(y / x); },          // BinOp::DivBy
    [](U, V y)   -> T { return static_cast<T>(y); },              // BinOp::Pass
    [](U x, V y) -> T { return static_cast<T>(x > y ? x : y); },  // BinOp::Max
    [](U x, V y) -> T { return static_cast<T>(x < y ? x : y); },  // BinOp::Min
    [](U x, V y) -> T { return static_cast<T>(x == y); },         // BinOp::Eq
    [](U x, V y) -> T { return static_cast<T>(x < y); },          // BinOp::Lt
    [](U x, V y) -> T { return static_cast<T>(x <= y); },         // BinOp::Lte
    [](U x, V y) -> T { return static_cast<T>(x > y); },          // BinOp::Gt
    [](U x, V y) -> T { return static_cast<T>(x >= y); },         // BinOp::Gte
};

template <typename T, typename U>
using UnOpFn = T(*)(U);

template <typename T, typename U>
static constexpr UnOpFn<T, U> unop_table[] = {
    [](U x) { return static_cast<T>(std::exp(x)); },
    [](U x) { return static_cast<T>(std::log(x)); },
    [](U x) { return static_cast<T>(-x); },
    [](U x) { return static_cast<T>(x); },
};

// template <typename T, typename U, typename KernelGen>
// constexpr auto make_kernel_table_unop(KernelGen gen) {
//     constexpr size_t N = sizeof(unop_table<T, U>) / sizeof(UnOpFn<T, U>);
//     return make_kernel_table_impl(gen, std::make_index_sequence<N>{});
// }

template <typename U>
using ArgRedOpFn = void(*)(std::pair<U, size_t>&, std::pair<U, size_t>);

template <typename U>
static constexpr ArgRedOpFn<U> argredop_table[] = {
    [](std::pair<U, size_t>& x, std::pair<U, size_t> y) { x = std::max(x, y); },
    [](std::pair<U, size_t>& x, std::pair<U, size_t> y) { x = std::min(x, y); },
};

template <typename U>
struct StridedIterator {
    U* data;
    Shape shape;
    Strides strides;
    std::array<size_t, config::MAX_DIMS> idxs;
    size_t dataIdx;
    size_t flatIdx;

    StridedIterator(U* data, const Shape& shape, const Strides& strides, size_t offset) 
        : data(data), shape(shape), strides(strides), idxs{}, dataIdx(offset), flatIdx(0) {}

    U& operator*() {
        return data[dataIdx];
    }

    StridedIterator& operator++() {
        for (int i = shape.size() - 1; i >= 0; i--) {
            dataIdx += strides[i];
            if (++idxs[i] == shape[i]) {
                idxs[i] = 0;
                dataIdx -= strides[i] * shape[i];
            }
            else break;
        }
        flatIdx++;
        return *this;
    }

    StridedIterator& operator+=(size_t n) {
        flatIdx += n;
        
        if (shape.size() == 0)
            return *this;

        idxs[shape.size() - 1] += n;
        dataIdx += n * strides[shape.size() - 1];

        for (int i = shape.size() - 1; i >= 1; i--) {
            if (idxs[i] >= shape[i]) {
                size_t num = idxs[i] / shape[i];
                idxs[i] -= num * shape[i];
                dataIdx -= num * shape[i] * strides[i];
                idxs[i - 1] += num;
                dataIdx += num * strides[i - 1];
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
        return other.data == data && other.dataIdx == dataIdx;
    }
};

}

}

#endif // BACKEND_CPU_UTIL_H