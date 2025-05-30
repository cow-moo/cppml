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

// Length of -1 means index the rest of the dimension.
struct Range {
    int start, length, step;

    Range() : start(0), length(-1), step(1) {}
    Range(int length) : start(0), length(length), step(1) {}
    Range(int start, int length, int step = 1) : start(start), length(length), step(step) {}

    friend std::ostream& operator<<(std::ostream& os, const Range& t) {
        os << "Range(" << t.start << ", " << t.length << ", " << t.step << ")";
        return os;
    }
};

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
    
    // Constructor for 1D tensor
    Tensor(const std::initializer_list<U>& values) : data(new U[values.size()]) {
        memcpy(data.get(), values.begin(), values.size() * sizeof(U));
        shape = {values.size()};
        strides = {1};
    }

    // Recursive constructor for arbitrary-dimensional tensors
    Tensor(const std::initializer_list<Tensor<U>>& slices) {
        if (slices.size() == 0) {
            throw std::invalid_argument("Tensor cannot be empty");
        }

        // Determine the shape of the first slice
        shape = slices.begin()->shape;
        strides = slices.begin()->strides;

        size_t subSize = strides.front() * shape.front();

        data = std::shared_ptr<U[]>(new U[slices.size() * subSize]);

        U* p = data.get();
        // Check consistency of shapes across slices
        for (const Tensor<U>& slice : slices) {
            if (slice.shape != shape) {
                throw std::invalid_argument("All slices must have the same shape");
            }
            memcpy(p, slice.data.get(), subSize * sizeof(U));
            p += subSize;
        }

        shape.insert(shape.begin(), slices.size());
        strides.insert(strides.begin(), subSize);
    }

    static Tensor zeros(const std::vector<size_t>& shape) {
        Tensor res;
        res.shape = shape;
        size_t totalSize = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            res.strides.push_back(totalSize);
            totalSize *= shape[i];
        }
        reverse(res.strides.begin(), res.strides.end());
        res.data = std::shared_ptr<U[]>(new U[totalSize]);
        std::fill(res.data.get(), res.data.get() + totalSize, 0);
        return res;
    }

    static Tensor normal(const std::vector<size_t>& shape, U mean = 0, U std = 1, std::optional<uint> seed = std::nullopt) {
        Tensor res;
        res.shape = shape;
        size_t totalSize = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            res.strides.push_back(totalSize);
            totalSize *= shape[i];
        }
        reverse(res.strides.begin(), res.strides.end());
        res.data = std::shared_ptr<U[]>(new U[totalSize]);

        std::mt19937 generator(seed ? *seed : std::random_device{}());
        std::normal_distribution<U> distribution(mean, std);
        for (size_t i = 0; i < totalSize; i++) {
            res.data[i] = distribution(generator);
        }
        return res;
    }

    // Performs a deep copy that discards data not seen/accessed by the view
    Tensor copy() const {
        Tensor res = zeros(shape);
        res += *this;
        return res;
    }

    // Variadic template access operator
    template <typename... Args>
    TensorProxy<U> operator[](Args... args) const {
        return TensorProxy<U>(*this, std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
    }

    // Apply some elementwise operation on two tensors with broadcasting
    // Func should have some signature op(U, U) -> U
    // Consider using template <auto op> with constexpr op instead of passing as parameter
    template <typename Func>
    Tensor apply_binary(const Tensor& other, Func op) const {
        std::vector<size_t> shape = broadcast_shape(this->shape, other.shape);

        Tensor res = zeros(shape);
        res.apply_binary_helper((*this), other, op, 0, indexOffset, other.indexOffset, 0);
        return res;
    }

    // Apply some elementwise operation on two tensors with broadcasting, modifying the LHS tensor in place
    // LHS tensor must be greater in all dimensions
    // Func should have some signature op(U&, U) -> void
    template <typename Func>
    Tensor& apply_binary_inplace(const Tensor& other, Func op) {
        if (shape != broadcast_shape(this->shape, other.shape)) {
            throw std::invalid_argument("Broadcast failed.");
        }

        apply_binary_inplace_helper(other, op, indexOffset, other.indexOffset, 0);
        return *this;
    }

    // Apply some operation on some tensor, returning a new one
    // Func should have signature op(U) -> void
    template <typename Func>
    Tensor apply_unary(Func op) const {
        Tensor res = zeros(shape);
        res.apply_unary_helper(*this, op, 0, indexOffset, 0);
        return res;
    }

    // Apply some operation on some tensor, modifying it in place
    // Func should have signature op(U&) -> void
    template <typename Func>
    Tensor& apply_unary_inplace(Func op) {
        apply_unary_inplace_helper(op, indexOffset, 0);
        return *this;
    }

    Tensor operator-() const {
        return *this * (U)-1;
    }

    Tensor operator+(const Tensor& other) const {
        return apply_binary(other, [](U a, U b) { return a + b; });
    }

    Tensor operator-(const Tensor& other) const {
        return apply_binary(other, [](U a, U b) { return a - b; });
    }

    Tensor operator*(const Tensor& other) const {
        return apply_binary(other, [](U a, U b) { return a * b; });
    }

    Tensor operator/(const Tensor& other) const {
        return apply_binary(other, [](U a, U b) { return a / b; });
    }

    Tensor operator+(U other) const {
        return apply_unary([other](U a) { return a + other; });
    }

    Tensor operator-(U other) const {
        return apply_unary([other](U a) { return a - other; });
    }

    Tensor operator*(U other) const {
        return apply_unary([other](U a) { return a * other; });
    }

    Tensor operator/(U other) const {
        return apply_unary([other](U a) { return a / other; });
    }

    Tensor& operator+=(const Tensor& other) {
        apply_binary_inplace(other, [](U& a, U b) { a += b; });
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        apply_binary_inplace(other, [](U& a, U b) { a -= b; });
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        apply_binary_inplace(other, [](U& a, U b) { a *= b; });
        return *this;
    }

    Tensor& operator/=(const Tensor& other) {
        apply_binary_inplace(other, [](U& a, U b) { a /= b; });
        return *this;
    }

    Tensor& operator+=(U other) {
        apply_unary_inplace([other](U& a) { a += other; });
        return *this;
    }

    Tensor& operator-=(U other) {
        apply_unary_inplace([other](U& a) { a -= other; });
        return *this;
    }

    Tensor& operator*=(U other) {
        apply_unary_inplace([other](U& a) { a *= other; });
        return *this;
    }

    Tensor& operator/=(U other) {
        apply_unary_inplace([other](U& a) { a /= other; });
        return *this;
    }

    Tensor sum() const {
        std::vector<int> axes(shape.size());
        for (size_t i = 0; i < shape.size(); i++) axes[i] = i;
        return sum(std::move(axes));
    }

    Tensor sum(int axis) const {
        return sum(std::vector<int>{axis});
    }
    
    // Allows duplicates and ignores them
    Tensor sum(std::vector<int>&& axes) const {
        if (axes.size() == 0)
            return copy();

        std::vector<bool> reduceAxis(shape.size(), false);

        for (auto x : axes) {
            if (x < 0) x += shape.size();
            if (x < 0 || x >= (int)shape.size()) {
                throw std::invalid_argument("Axis index out of bounds.");
            }

            reduceAxis[x] = true;
        }
        
        std::vector<size_t> newShape;
        for (size_t i = 0; i < shape.size(); i++) {
            if (!reduceAxis[i])
                newShape.push_back(shape[i]);
        }

        Tensor res = zeros(newShape);
        res.sum_helper(*this, reduceAxis, 0, indexOffset, 0, 0);
        return res;
    }

    // Broadcasts first n-2 dimensions
    friend Tensor matmul(const Tensor& a, const Tensor& b) {
        // Prepend 1 to shape of a if 1D
        Tensor aa = a.shape.size() < 2 ? a.reshape({1, a.shape[0]}) : a;
        // Append 1 to shape of b if 1D
        Tensor bb = b.shape.size() < 2 ? b.reshape({b.shape[0], 1}) : b;

        if (aa.shape.back() != bb.shape[bb.shape.size() - 2]) {
            throw std::invalid_argument("Invalid shape for matmul.");
        }
        std::vector<size_t> shape = broadcast_shape(std::vector<size_t>(aa.shape.begin(), aa.shape.begin() + aa.shape.size() - 2), 
                                                    std::vector<size_t>(bb.shape.begin(), bb.shape.begin() + bb.shape.size() - 2));
        shape.push_back(aa.shape[aa.shape.size() - 2]);
        shape.push_back(bb.shape.back());

        Tensor res = zeros(shape);
        res.matmul_helper(aa, bb, 0, aa.indexOffset, bb.indexOffset, 0);
        if (a.shape.size() < 2) {
            // Remove 1 if a was reshaped
            assert(res.shape[res.shape.size() - 2] == 1);
            res.strides.erase(res.strides.end() - 2);
            res.shape.erase(res.shape.end() - 2);
        }
        if (b.shape.size() < 2) {
            // Remove 1 if b was reshaped
            assert(res.shape.back() == 1);
            res.strides.pop_back();
            res.shape.pop_back();
        }
        return res;
    }

    Tensor T() const {
        Tensor res(*this);
        std::reverse(res.strides.begin(), res.strides.end());
        std::reverse(res.shape.begin(), res.shape.end());
        return res; // Should do NRVO
    }

    Tensor reshape(const std::vector<size_t>& newShape) const {
        // Could modify to handle this case if necessary
        if (shape.size() == 0 || newShape.size() == 0) {
            throw std::invalid_argument("Invalid shape for reshape.");
        }
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        for (auto dim : newShape) {
            if (size % dim != 0) {
                throw std::invalid_argument("Invalid shape for reshape.");
            }
            size /= dim;
        }
        if (size != 1) {
            throw std::invalid_argument("Invalid shape for reshape.");
        }

        size_t cur = 1;
        size_t sizeCur = shape[0];
        size_t lNew = 0, rNew = 0;
        size_t sizeNew = 1;

        std::vector<size_t> newStrides;

        // Logic is still a little sus but seems to work
        bool full = true;
        while (lNew < newShape.size()) {
            if (sizeCur < sizeNew && cur < shape.size()) {
                if (strides[cur - 1] != strides[cur] * shape[cur])
                    full = false;
                sizeCur *= shape[cur++];
            }
            else if (rNew < newShape.size()) {
                sizeNew *= newShape[rNew++];
            }

            if (sizeCur == sizeNew) {
                if (!full) {
                    // Need to make a copy
                    std::cout << "reshape copy made" << std::endl;
                    return copy().reshape(newShape);
                }

                int stride = strides[cur - 1];
                for (; lNew < rNew; lNew++) {
                    sizeNew /= newShape[lNew];
                    newStrides.push_back(sizeNew * stride);
                }
                assert(sizeNew == 1);

                if (cur == shape.size())
                    sizeCur = 1;
                else
                    sizeCur = shape[cur++];
                full = true;
            }
        }

        Tensor res(*this);
        res.strides = newStrides;
        res.shape = newShape;

        return res;
    }

    //Cast Tensors with no dimension to scalar, implicit cast for cout/math
    operator U() const {
        if (shape.size() > 0) {
            throw std::invalid_argument("Can't cast tensor to scalar.");
        }
        return data[indexOffset];
    }

    // Define << for printing using recursion
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

    void print() const {
        std::cout << "(";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << shape[i];
            if (i < shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "): " << (*this) << std::endl;
    }

    void print_shape() const {
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    Tensor broadcast_reduce_to(std::vector<size_t> shape) {
        std::vector<int> broadcastedAxes;
        for (size_t i = 0; i < this->shape.size(); i++) {
            if (i >= shape.size() || this->shape[this->shape.size() - 1 - i] != shape[shape.size() - 1 - i]) {
                broadcastedAxes.push_back(this->shape.size() - 1 - i);
            }
        }

        return sum(std::move(broadcastedAxes));
    }

protected:
    std::shared_ptr<U[]> data;
    std::vector<size_t> strides;
    size_t indexOffset = 0;

    Tensor() {}

    // Write element wise op(a, b) into this object
    // Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
    template <typename Func>
    void apply_binary_helper(const Tensor& a, const Tensor& b, Func op, int index, int aIndex, int bIndex, int axis) {
        if (axis == (int)shape.size()) {
            data[index] = op(a.data[aIndex], b.data[bIndex]);
            return;
        }

        int aAxis = axis + a.shape.size() - shape.size();
        if (aAxis >= 0 && a.shape[aAxis] == 1) {
            aAxis = -1;
        }
        
        int bAxis = axis + b.shape.size() - shape.size();
        if (bAxis >= 0 && b.shape[bAxis] == 1) {
            bAxis = -1;
        }

        for (size_t i = 0; i < shape[axis]; i++) {
            apply_binary_helper(
                a, b, op, 
                index + strides[axis] * i, 
                aIndex + (aAxis >= 0 ? a.strides[aAxis] * i : 0),
                bIndex + (bAxis >= 0 ? b.strides[bAxis] * i : 0),
                axis + 1
            );
        }
    }

    // Write element wise op(a, b) into this object
    // Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
    template <typename Func>
    void apply_binary_inplace_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
        if (axis == (int)shape.size()) {
            op(data[index], other.data[otherIndex]);
            return;
        }
        
        int otherAxis = axis + other.shape.size() - shape.size();
        if (otherAxis >= 0 && other.shape[otherAxis] == 1) {
            otherAxis = -1;
        }

        for (size_t i = 0; i < shape[axis]; i++) {
            apply_binary_inplace_helper(
                other, op, 
                index + strides[axis] * i, 
                otherIndex + (otherAxis >= 0 ? other.strides[otherAxis] * i : 0),
                axis + 1
            );
        }
    }

    // Apply op elementwise to other and copy into this. Guaranteed that this tensor and other are the same shape
    template <typename Func>
    void apply_unary_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
        if (axis == shape.size()) {
            data[index] = op(other.data[otherIndex]);
            return;
        }

        for (size_t i = 0; i < shape[axis]; i++) {
            apply_unary_helper(other, op, index + strides[axis] * i, otherIndex + other.strides[axis] * i, axis + 1);
        }
    }

    // Apply elementwise
    template <typename Func>
    void apply_unary_inplace_helper(Func op, int index, int axis) {
        if (axis == (int)shape.size()) {
            op(data[index]);
            return;
        }

        for (size_t i = 0; i < shape[axis]; i++) {
            apply_unary_inplace_helper(op, index + strides[axis] * i, axis + 1);
        }
    }

    void matmul_helper(const Tensor& a, const Tensor& b, int index, int aIndex, int bIndex, int axis) {
        int aAxis = axis + a.shape.size() - shape.size();
        int bAxis = axis + b.shape.size() - shape.size();
        
        if (axis == (int)shape.size() - 2) {
            // Actual matmul here
            for (size_t i = 0; i < shape[axis]; i++) {
                for (size_t j = 0; j < shape[axis + 1]; j++) {
                    for (size_t k = 0; k < a.shape[aAxis + 1]; k++) {
                        data[index + strides[axis] * i + strides[axis + 1] * j] += 
                            a.data[aIndex + a.strides[aAxis] * i + a.strides[aAxis + 1] * k] * 
                            b.data[bIndex + b.strides[bAxis] * k + b.strides[bAxis + 1] * j];
                    }
                }
            }
            return;
        }

        if (aAxis >= 0 && a.shape[aAxis] == 1) {
            aAxis = -1;
        }

        if (bAxis >= 0 && b.shape[bAxis] == 1) {
            bAxis = -1;
        }

        for (size_t i = 0; i < shape[axis]; i++) {
            matmul_helper(
                a, b, 
                index + strides[axis] * i, 
                aIndex + (aAxis >= 0 ? a.strides[aAxis] * i : 0),
                bIndex + (bAxis >= 0 ? b.strides[bAxis] * i : 0),
                axis + 1
            );
        }
    }

    void sum_helper(const Tensor& other, const std::vector<bool>& reduceAxis, int index, int otherIndex, int axis, int otherAxis) const {
        if (otherAxis == (int)other.shape.size()) {
            data[index] += other.data[otherIndex];
            return;
        }        

        if (reduceAxis[otherAxis]) {
            for (size_t i = 0; i < other.shape[otherAxis]; i++) {
                sum_helper(other, reduceAxis, index, otherIndex + other.strides[otherAxis] * i, axis, otherAxis + 1);
            }
        }
        else {
            assert(shape[axis] == other.shape[otherAxis]);
            for (size_t i = 0; i < other.shape[otherAxis]; i++) {
                sum_helper(other, reduceAxis, index + strides[axis] * i, otherIndex + other.strides[otherAxis] * i, axis + 1, otherAxis + 1);
            }
        }
    }

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        std::vector<size_t> res;
        for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
            if (i >= a.size()) {
                res.push_back(b[b.size() - 1 - i]);
            }
            else if (i >= b.size()) {
                res.push_back(a[a.size() - 1 - i]);
            }
            else if (a[a.size() - 1 - i] == 1) {
                res.push_back(b[b.size() - 1 - i]);
            }
            else if (b[b.size() - 1 - i] == 1) {
                res.push_back(a[a.size() - 1 - i]);
            }
            else if (a[a.size() - 1 - i] == b[b.size() - 1 - i]) {
                res.push_back(a[a.size() - 1 - i]);
            }
            else {
                throw std::invalid_argument("Broadcast failed.");
                //return {};
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

template <typename U>
Tensor<U> operator+(U a, const Tensor<U>& b) {
    return b + a;
}

template <typename U>
Tensor<U> operator-(U a, const Tensor<U>& b) {
    return b.apply_unary([a](U b) { return a - b; });
}

template <typename U>
Tensor<U> operator*(U a, const Tensor<U>& b) {
    return b * a;
}

template <typename U>
Tensor<U> operator/(U a, const Tensor<U>& b) {
    return b.apply_unary([a](U b) { return a / b; });
}

// Almost all Tensor operations should return TensorProxy, but it seems to break something and it doesn't seem important enough to fix
template <typename U>
class TensorProxy : public Tensor<U> {
public:
    using Tensor<U>::shape;

    template <std::size_t... Is, typename... Args>
    TensorProxy(const Tensor<U>& orig, std::index_sequence<Is...>, Args&&... args) {
        data = orig.data;
        indexOffset = orig.indexOffset;
        // Fold expression to process all arguments with overloads
        ((process_get_arg(orig, Is, args)), ...);
        
        // Copy remaining dimensions
        for (size_t i = sizeof...(Args); i < orig.strides.size(); i++) {
            strides.push_back(orig.strides[i]);
            shape.push_back(orig.shape[i]);
        }
    }

    TensorProxy& operator=(U other) {
        Tensor<U>::apply_unary_inplace([other](U& a) { a = other; });
        return *this;
    }

    TensorProxy& operator=(const Tensor<U>& other) {
        Tensor<U>::apply_binary_inplace(other, [](U& a, U b) { a = b; });
        return *this;
    }

protected:
    using Tensor<U>::data;
    using Tensor<U>::strides;
    using Tensor<U>::indexOffset;

    void process_get_arg(const Tensor<U>& orig, std::size_t idx, int x) {
        if (idx >= orig.shape.size()) {
            throw std::invalid_argument("Too many arguments provided for get.");
        }
        // Allow Python-style negative indexing
        if (x < 0) {
            x += orig.shape[idx];
        }
        if (x >= (int)orig.shape[idx] || x < 0) {
            throw std::invalid_argument("Index out of bounds.");
        }
        // Add to indexOffset and don't add a range to axisIndices, reducing number of axes
        indexOffset += orig.strides[idx] * x;
    }

    void process_get_arg(const Tensor<U>& orig, std::size_t idx, Range range) {
        if (idx >= orig.strides.size()) {
            throw std::invalid_argument("Too many arguments provided for get.");
        }
        
        // Allow Python-style negative indexing
        if (range.start < 0) {
            range.start += orig.shape[idx];
        }
        if (range.start < 0 || range.start >= (int)orig.shape[idx]) {
            throw std::invalid_argument("Index out of bounds.");
        }

        // Length of -1 implies index to the end
        if (range.length == -1) {
            range.length = range.step > 0 ? (orig.shape[idx] - range.start + range.step - 1) / range.step : (range.start / -range.step) + 1;
        }
        if (range.length <= 0) {
            throw std::invalid_argument("Negative length.");
        }
       
        shape.push_back(range.length);
        strides.push_back(orig.strides[idx] * range.step);
        indexOffset += orig.strides[idx] * range.start;
    }
};

} // namespace linalg

#endif // TENSOR_H


/*
Future optimizations:
- store backprop functions in arena instead of capturing lambdas to avoid heap allocations
- make data into an intrusive pointer to avoid an indirection


do proper tensor/tensorview with ownership so that we can be confident we're storing data
make tensors own data
*/