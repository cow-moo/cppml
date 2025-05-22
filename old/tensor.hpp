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

namespace linalg {

template <typename T>
concept Iterable = requires(T t) {
    std::begin(t);
    std::end(t);
    t.size();
};

template <typename T>
concept IterableOfInt = Iterable<T> && (requires(T t, std::size_t i) {
    { t[i] } -> std::same_as<int>;
} || requires(T t, std::size_t i) {
    { t[i] } -> std::same_as<int&>;
});

template <typename T>
concept IntOrIterableOfInt = std::same_as<T, int> || std::same_as<T, size_t> || IterableOfInt<T>;

// Length of 0 means index the rest of the dimension.
class Range {
public:
    class Iterator {
    public:
        Iterator(int current, int step) : current(current), step(step) {}
        int operator*() const { return current; }
        Iterator& operator++() { current += step; return *this; }
        bool operator!=(const Iterator& other) const { return step > 0 ? current < other.current : current > other.current; }
    private:
        int current, step;
    };

    Range() : start(0), length(0), step(1) {}
    Range(int length) : start(0), length(length), step(1) {}
    Range(int start, int length, int step = 1) : start(start), length(length), step(step) {}

    Iterator begin() const { return Iterator(start, step); }
    Iterator end() const { return Iterator(start + length * step, step); }

    int operator[](int i) const;

    // Range composition
    Range operator[](const Range& other) const;

    int size() const { return length; }

    friend std::ostream& operator<<(std::ostream& os, const Range& t);

    int start, length, step;
};

// TODO implement this
// struct Shape {
//     std::vector<size_t> data;

//     size_t operator[](int i) { return data[i]; }
// };

template <typename U = float>
class TensorProxy;

template <typename U = float>
class Tensor {
public:
    // Should be fixed on construction
    std::vector<size_t> shape;

    friend class TensorProxy<U>;
    
    Tensor(std::initializer_list<U> values) : Tensor(std::vector<U>(values)) {}
    Tensor(std::initializer_list<Tensor<U>> slices) : Tensor(std::vector<Tensor<U>>(slices)) {}

    // Constructor for 1D tensor
    Tensor(std::vector<U>&& values);

    // Recursive constructor for arbitrary-dimensional tensors
    Tensor(std::vector<Tensor<U>>&& slices);

    //Tensor(U value) : data(std::make_shared<std::vector<U>>({value})) {}

    static Tensor zeros(const std::vector<size_t>& shape);

    static Tensor normal(const std::vector<size_t>& shape, U mean = 0, U std = 1, std::optional<uint> seed = std::nullopt);

    // Performs a deep copy that discards data not seen/accessed by the view
    // Maybe try to remove recursion? incurs extra factor of #axes data copying
    // Switch to using apply unary?
    Tensor copy() const;

    // Variadic template access operator
    template <IntOrIterableOfInt... Args>
    TensorProxy<U> operator[](Args... args) const;

    // Apply some elementwise operation on two tensors with broadcasting
    // Func should have some signature op(U, U) -> U
    // Consider using template <auto op> with constexpr op instead of passing as parameter
    template <typename Func>
    Tensor apply_binary(const Tensor& other, Func op) const;

    // Apply some elementwise operation on two tensors with broadcasting, modifying the LHS tensor in place
    // LHS tensor must be greater in all dimensions
    // Func should have some signature op(U&, U) -> void
    template <typename Func>
    Tensor& apply_binary_inplace(const Tensor& other, Func op);

    // Apply some operation on some tensor, returning a new one
    // Func should have signature op(U) -> void
    template <typename Func>
    Tensor apply_unary(Func op) const;

    // Apply some operation on some tensor, modifying it in place
    // Func should have signature op(U&) -> void
    template <typename Func>
    Tensor& apply_unary_inplace(Func op);

    Tensor operator-() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(U other) const;
    Tensor operator-(U other) const;
    Tensor operator*(U other) const;
    Tensor operator/(U other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor& operator+=(U other);
    Tensor& operator-=(U other);
    Tensor& operator*=(U other);
    Tensor& operator/=(U other);

    Tensor sum();
    Tensor sum(int axis);
    Tensor sum(std::vector<int>&& axes);

    // Broadcasts first n-2 dimensions
    static Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor T() const;

    Tensor reshape(const std::vector<size_t>& newShape) const;

    //Cast Tensors with no dimension to scalar, implicit cast for cout/math
    operator U() const;

    // Define << for printing using recursion
    //friend std::ostream& operator<< <>(std::ostream& os, const Tensor<U>& t);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (t.shape.size() == 0) {
            os << ((U)t);
            return os;
        }
        os << "[";
        for (size_t i = 0; i < t.shape[0]; i++) {
            os << t[i];
            if (i < t.shape[0] - 1)
                os << ", ";
        }
        os << "]";

        return os;
    }

    void print() const;
    void print_axis_indices();
    void print_shape() const;

protected:
    std::shared_ptr<std::vector<U>> data;
    std::vector<Range> axisIndices;
    size_t indexOffset = 0;

    Tensor() {}

    template <typename Func>
    void apply_binary_helper(const Tensor& a, const Tensor& b, Func op, int index, int aIndex, int bIndex, int axis);

    template <typename Func>
    void apply_binary_inplace_helper(const Tensor& other, Func op, int index, int otherIndex, int axis);
    
    template <typename Func>
    void apply_unary_helper(const Tensor& other, Func op, int index, int otherIndex, int axis);

    template <typename Func>
    void apply_unary_inplace_helper(Func op, int index, int axis);

    void matmul_helper(const Tensor& a, const Tensor& b, int index, int aIndex, int bIndex, int axis);
    void sum_helper(const Tensor& other, const std::vector<bool>& reduceAxis, int index, int otherIndex, int axis, int otherAxis);
};

template <typename U>
Tensor<U> operator+(U a, const Tensor<U>& b);

template <typename U>
Tensor<U> operator-(U a, const Tensor<U>& b);

template <typename U>
Tensor<U> operator*(U a, const Tensor<U>& b);

template <typename U>
Tensor<U> operator/(U a, const Tensor<U>& b);

// Almost all Tensor operations should return TensorProxy, but it seems to break something and it doesn't seem important enough to fix
template <typename U>
class TensorProxy : public Tensor<U> {
public:
    using Tensor<U>::shape;

    template <std::size_t... Is, IntOrIterableOfInt... Args>
    TensorProxy(const Tensor<U>& orig, std::index_sequence<Is...>, Args&&... args);

    // Elementwise assignment
    TensorProxy& operator=(U other);
    TensorProxy& operator=(const Tensor<U>& other);

protected:
    using Tensor<U>::data;
    using Tensor<U>::axisIndices;
    using Tensor<U>::indexOffset;

    void process_get_arg(const Tensor<U>& orig, std::size_t idx, int x);
    void process_get_arg(const Tensor<U>& orig, std::size_t idx, const Range& range);
};

} // namespace linalg

#endif // TENSOR_H