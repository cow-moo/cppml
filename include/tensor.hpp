#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <concepts>
#include <vector>
#include <memory>
#include <variant>
#include <algorithm>
#include <sstream>
#include <span>
#include <cassert>
#include <random>
#include <optional>
#include "shape.hpp"
#include "config.hpp"
#include "backend.hpp"

namespace linalg {

// Length of -1 means index the rest of the dimension.
struct Range {
    int start, length, step;
    Range();
    Range(int length);
    Range(int start, int length, int step = 1);
    friend std::ostream& operator<<(std::ostream& os, const Range& t);
};

template <typename U = float>
class Tensor {
public:
    struct NestedInitializer;

    /* ===== Constructors & assignment ===== */
    Tensor(const NestedInitializer& initializer, backend::BackendType type = backend::current_backend_type);
    Tensor(const Shape& shape, backend::BackendType type = backend::current_backend_type);
    Tensor(const std::initializer_list<U>& list, backend::BackendType type = backend::current_backend_type);
    Tensor(const Tensor& other);
    Tensor& operator=(U other);
    Tensor& operator=(const Tensor& other);
    Tensor& assign(const Tensor& other);

    /* ===== Static factories ===== */
    static Tensor zeros(const Shape& shape, backend::BackendType type = backend::current_backend_type);
    static Tensor normal(const Shape& shape, U mean, U std, std::optional<uint> seed, backend::BackendType type = backend::current_backend_type);
    // Uses N(0, 1)
    static Tensor normal(const Shape& shape, backend::BackendType type = backend::current_backend_type);

    /* ===== Copy & indexing ===== */
    Tensor copy() const;
    template <typename... Args>
    Tensor operator[](Args&&... args) const;
    template <typename... Args>
    Tensor at(Args&&... args) const;

    /* ===== Arithmetic ===== */
    template <typename V>
    friend Tensor<V> operator+(const Tensor<V>& a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<V> operator-(const Tensor<V>& a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<V> operator*(const Tensor<V>& a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<V> operator/(const Tensor<V>& a, const Tensor<V>& b);

    Tensor operator+(U other) const;
    Tensor operator-(U other) const;
    Tensor operator*(U other) const;
    Tensor operator/(U other) const;

    template <typename V>
    friend Tensor<V> operator-(V a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<V> operator/(V a, const Tensor<V>& b);

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor& operator+=(U other);
    Tensor& operator-=(U other);
    Tensor& operator*=(U other);
    Tensor& operator/=(U other);

    /* ===== Comparison ===== */
    template <typename V>
    friend Tensor<bool> operator==(const Tensor<V>& a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<bool> operator<(const Tensor<V>& a, const Tensor<V>& b);
    template <typename V>
    friend Tensor<bool> operator<=(const Tensor<V>& a, const Tensor<V>& b);

    Tensor<bool> operator<(U other) const;
    Tensor<bool> operator>(U other) const;
    Tensor<bool> operator<=(U other) const;
    Tensor<bool> operator>=(U other) const;
    Tensor<bool> operator==(U other) const;

    /* ===== Unary Ops ===== */
    Tensor operator-() const;
    Tensor exp() const;
    Tensor& exp_();
    Tensor log() const;
    Tensor& log_();
    Tensor neg() const;
    Tensor& neg_();
    Tensor relu() const;

    /* ===== Reductions ===== */
    Tensor sum(const std::vector<int>& axes, bool keepDims = false) const;
    Tensor sum() const;
    Tensor max(const std::vector<int>& axes, bool keepDims = false) const;
    Tensor min(const std::vector<int>& axes, bool keepDims = false) const;
    Tensor<size_t> argmax(int axis = -1, bool keepDim = false) const;
    Tensor<size_t> argmin(int axis = -1, bool keepDim = false) const;
    Tensor softmax(int axis = -1) const;
    Tensor log_softmax(int axis = -1) const;
    Tensor broadcast_reduce_to(const Shape& shape);

    /* ===== Linear Algebra ===== */
    template <typename V>
    friend Tensor<V> matmul(const Tensor<V>& a, const Tensor<V>& b);

    /* ===== Manipulations ===== */
    Tensor reshape(const Shape& newShape) const;
    Tensor unsqueeze(int axis) const;
    Tensor squeeze(const std::vector<int>& axes) const;
    Tensor T() const;

    /* ===== Info ===== */
    size_t numel() const;
    const Shape& shape() const;
    backend::BackendType backend_type() const;

    /* ===== Type / Backend Casting ===== */
    template <typename R>
    Tensor<R> astype() const;
    Tensor to(backend::BackendType type) const;
    operator U() const;

    /* ===== Debug ===== */
    template <typename V>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<V>& t);
    void print() const;
    void print_shape() const;

    struct NestedInitializer {
        std::vector<U> flatData;
        Shape shape;
        NestedInitializer(const std::initializer_list<U>& slice);
        NestedInitializer(const std::initializer_list<NestedInitializer>& slices);
        NestedInitializer(const std::vector<U>& flat);
    };

private:
    backend::SharedBuffer<U> data_;
    Shape shape_;
    Strides strides_;
    size_t offset_ = 0;

    template <typename>
    friend class Tensor;

    /* ===== Internal Helpers ===== */
    template <std::size_t... Is, typename... Args>
    Tensor(const Tensor& orig, std::index_sequence<Is...>, Args&&... args);
    void process_get_arg(const Tensor& orig, std::size_t idx, int x);
    void process_get_arg(const Tensor& orig, std::size_t idx, Range range);
    template <typename R = U, typename V = U>
    Tensor<R> apply_binary(const Tensor<V>& other, backend::BinOp op) const;
    template <typename R = U, typename V = U>
    Tensor<R> apply_binary(V other, backend::BinOp op) const;
    template <typename V = U>
    Tensor& apply_binary_inplace(const Tensor<V>& other, backend::BinOp op);
    template <typename V = U>
    Tensor& apply_binary_inplace(V other, backend::BinOp op);
    template <typename R = U>
    Tensor<R> apply_unary(backend::UnOp op) const;
    Tensor& apply_unary_inplace(backend::UnOp op);
    Tensor reduce(const std::vector<int>& axes, U identity, backend::BinOp op, bool keepDims) const;
    Tensor<size_t> arg_reduce(int axis, backend::ArgRedOp op, bool keepDim) const;
};


// Non-member operator and utility function declarations
template <typename U>
Tensor<U> operator+(U a, const Tensor<U>& b);
template <typename U>
Tensor<U> operator*(U a, const Tensor<U>& b);

template <typename U>
Tensor<bool> operator==(U a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator>(const Tensor<U>& a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator>=(const Tensor<U>& a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator<(U a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator>(U a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator<=(U a, const Tensor<U>& b);
template <typename U>
Tensor<bool> operator>=(U a, const Tensor<U>& b);

template <typename U>
Tensor<U> exp(const Tensor<U>& t);
template <typename U>
Tensor<U> log(const Tensor<U>& t);
template <typename U>
Tensor<U> sum(const Tensor<U>& t);

}

#include "../src/tensor.tpp"

#endif // TENSOR_H