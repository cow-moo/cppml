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

template <typename U = float>
class Tensor {
public:
    // Constructor for 1D tensor
    Tensor(const std::initializer_list<U>& values) : data_(new U[values.size()]) {
        memcpy(data_.get(), values.begin(), values.size() * sizeof(U));
        shape_ = {values.size()};
        strides_ = {1};
    }

    // Recursive constructor for arbitrary-dimensional tensors
    Tensor(const std::initializer_list<Tensor<U>>& slices) {
        if (slices.size() == 0) {
            throw std::invalid_argument("Tensor cannot be empty");
        }

        // Determine the shape of the first slice
        shape_ = slices.begin()->shape_;
        strides_ = slices.begin()->strides_;

        size_t subSize = strides_.front() * shape_.front();

        data_ = std::shared_ptr<U[]>(new U[slices.size() * subSize]);

        U* p = data_.get();
        // Check consistency of shapes across slices
        for (const Tensor<U>& slice : slices) {
            if (slice.shape_ != shape_) {
                throw std::invalid_argument("All slices must have the same shape");
            }
            memcpy(p, slice.data_.get(), subSize * sizeof(U));
            p += subSize;
        }

        shape_.insert(shape_.begin(), slices.size());
        strides_.insert(strides_.begin(), subSize);
    }

    Tensor(const Tensor& other) {
        assign(other);
    }

    // Elementwise assignment
    Tensor& operator=(U other) {
        apply_unary_inplace([other](U& a) { a = other; });
        return *this;
    }

    // Elementwise assignment
    Tensor& operator=(const Tensor& other) {
        apply_binary_inplace(other, [](U& a, U b) { a = b; });
        return *this;
    }

    // Rebinding assignment
    Tensor& assign(const Tensor& other) {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        offset_ = other.offset_;
        return *this;
    }

    static Tensor zeros(const Shape& shape) {
        Tensor res(shape);
        res = 0;
        //std::fill(res.data_.get(), res.data_.get(), 0);
        return res;
    }

    static Tensor normal(const Shape& shape, U mean = 0, U std = 1, std::optional<uint> seed = std::nullopt) {
        Tensor res(shape);

        std::mt19937 generator(seed ? *seed : std::random_device{}());
        std::normal_distribution<U> distribution(mean, std);
        res.apply_unary_inplace([&](U& a) { a = distribution(generator); });

        return res;
    }

    // Performs a deep copy that discards data not seen/accessed by the view
    Tensor copy() const {
        Tensor res = zeros(shape_);
        res += *this;
        return res;
    }

    // Variadic template access operator
    template <typename... Args>
    Tensor operator[](Args&&... args) const {
        return Tensor(*this, std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
    }

    // Apply some elementwise operation on two tensors with broadcasting
    // Func should have some signature op(U, U) -> U
    // Consider using template <auto op> with constexpr op instead of passing as parameter
    template <typename Func>
    Tensor apply_binary(const Tensor& other, Func op) const {
        Shape shape = Shape::broadcast(this->shape_, other.shape_);

        Tensor res = zeros(shape);
        res.apply_binary_helper((*this), other, op, 0, offset_, other.offset_, 0);
        return res;
    }

    // Apply some elementwise operation on two tensors with broadcasting, modifying the LHS tensor in place
    // LHS tensor must be greater in all dimensions
    // Func should have some signature op(U&, U) -> void
    template <typename Func>
    Tensor& apply_binary_inplace(const Tensor& other, Func op) {
        if (shape_ != Shape::broadcast(this->shape_, other.shape_)) {
            throw std::invalid_argument("Broadcast failed.");
        }

        apply_binary_inplace_helper(other, op, offset_, other.offset_, 0);
        return *this;
    }

    // Apply some operation on some tensor, returning a new one
    // Func should have signature op(U) -> U
    template <typename Func>
    Tensor apply_unary(Func op) const {
        Tensor res = zeros(shape_);
        res.apply_unary_helper(*this, op, 0, offset_, 0);
        return res;
    }

    // Apply some operation on some tensor, modifying it in place
    // Func should have signature op(U&) -> void
    template <typename Func>
    Tensor& apply_unary_inplace(Func op) {
        apply_unary_inplace_helper(op, offset_, 0);
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
        std::vector<int> axes(shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) axes[i] = i;
        return sum(std::move(axes));
    }

    Tensor sum(int axis) const {
        return sum(std::vector<int>{axis});
    }
    
    // Allows duplicates and ignores them
    Tensor sum(std::vector<int>&& axes) const {
        if (axes.size() == 0)
            return copy();

        std::vector<bool> reduceAxis(shape_.size(), false);

        for (auto x : axes) {
            if (x < 0) x += shape_.size();
            if (x < 0 || x >= (int)shape_.size()) {
                throw std::invalid_argument("Axis index out of bounds.");
            }

            reduceAxis[x] = true;
        }
        
        Shape newShape;
        for (size_t i = 0; i < shape_.size(); i++) {
            if (!reduceAxis[i])
                newShape.push_back(shape_[i]);
        }

        Tensor res = zeros(newShape);
        res.sum_helper(*this, reduceAxis, 0, offset_, 0, 0);
        return res;
    }

    // Broadcasts first n-2 dimensions
    // 1D tensors have 1 appended (column vector)
    friend Tensor matmul(const Tensor& a, const Tensor& b) {
        // Append 1 to shape of a if 1D
        Tensor aa = a.shape_.size() < 2 ? a.reshape({a.shape_[0], 1}) : a;
        // Append 1 to shape of b if 1D
        Tensor bb = b.shape_.size() < 2 ? b.reshape({b.shape_[0], 1}) : b;

        if (aa.shape_.back() != bb.shape_[bb.shape_.size() - 2]) {
            std::cout << a.shape_ << " " << b.shape_ << std::endl;
            throw std::invalid_argument("Invalid shape for matmul.");
        }
        Shape shape = Shape::broadcast(std::vector<size_t>(aa.shape_.begin(), aa.shape_.begin() + aa.shape_.size() - 2), 
                                                    std::vector<size_t>(bb.shape_.begin(), bb.shape_.begin() + bb.shape_.size() - 2));
        shape.push_back(aa.shape_[aa.shape_.size() - 2]);
        shape.push_back(bb.shape_.back());

        Tensor res = zeros(shape);
        res.matmul_helper(aa, bb, 0, aa.offset_, bb.offset_, 0);
        // if (a.shape_.size() < 2) {
        //     // Remove 1 if a was reshaped
        //     assert(res.shape_[res.shape_.size() - 2] == 1);
        //     res.strides_.erase(res.strides_.end() - 2);
        //     res.shape_.erase(res.shape_.end() - 2);
        // }
        if (b.shape_.size() < 2) {
            // Remove 1 if b was reshaped
            assert(res.shape_.back() == 1);
            res.strides_.pop_back();
            res.shape_.pop_back();
        }
        return res;
    }

    Tensor T() const {
        if (shape_.size() < 2) {
            return reshape({1, shape_[0]});
        }

        Tensor res(*this);
        std::swap(res.shape_[res.shape_.size() - 1], res.shape_[res.shape_.size() - 2]);
        std::swap(res.strides_[res.strides_.size() - 1], res.strides_[res.strides_.size() - 2]);
        return res;
    }

    Tensor reshape(const Shape& newShape) const {
        // Could modify to handle this case if necessary
        if (shape_.size() == 0 || newShape.size() == 0) {
            throw std::invalid_argument("Invalid shape for reshape.");
        }
        size_t size = 1;
        for (auto dim : shape_) {
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
        size_t sizeCur = shape_[0];
        size_t lNew = 0, rNew = 0;
        size_t sizeNew = 1;

        std::vector<size_t> newStrides;

        // Logic is still a little sus but seems to work
        bool full = true;
        while (lNew < newShape.size()) {
            if (sizeCur < sizeNew && cur < shape_.size()) {
                if (strides_[cur - 1] != strides_[cur] * shape_[cur])
                    full = false;
                sizeCur *= shape_[cur++];
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

                int stride = strides_[cur - 1];
                for (; lNew < rNew; lNew++) {
                    sizeNew /= newShape[lNew];
                    newStrides.push_back(sizeNew * stride);
                }
                assert(sizeNew == 1);

                if (cur == shape_.size())
                    sizeCur = 1;
                else
                    sizeCur = shape_[cur++];
                full = true;
            }
        }

        Tensor res(*this);
        res.strides_ = newStrides;
        res.shape_ = newShape;

        return res;
    }

    size_t numel() {
        size_t res = 1;
        for (auto x : shape_) {
            res *= x;
        }
        return res;
    }

    // Cast Tensors with no dimension to scalar, implicit cast for cout/math
    operator U() const {
        if (shape_.size() > 0) {
            throw std::invalid_argument("Can't cast tensor to scalar.");
        }
        return data_[offset_];
    }

    // Define << for printing using recursion
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (t.shape_.size() == 0) {
            os << ((U)t);
            return os;
        }
        os << "[";
        for (size_t i = 0; i < t.shape_[0]; i++) {
            os << t[i];
            if (i < t.shape_[0] - 1)
                os << ", ";
        }
        os << "]";

        return os;
    }

    void print() const {
        std::cout << shape_;
        // std::cout << "(";
        // for (size_t i = 0; i < shape_.size(); i++) {
        //     std::cout << shape_[i];
        //     if (i < shape_.size() - 1)
        //         std::cout << ", ";
        // }
        std::cout << ": " << (*this) << std::endl;
    }

    void print_shape() const {
        for (auto dim : shape_) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    Tensor broadcast_reduce_to(const Shape& shape) {
        std::vector<int> broadcastedAxes;
        for (size_t i = 0; i < this->shape_.size(); i++) {
            if (i >= shape.size() || this->shape_[this->shape_.size() - 1 - i] != shape[shape.size() - 1 - i]) {
                broadcastedAxes.push_back(this->shape_.size() - 1 - i);
            }
        }

        return sum(std::move(broadcastedAxes));
    }

    const Shape& shape() const {
        return shape_;
    }

protected:
    // Should be fixed on construction
    std::shared_ptr<U[]> data_;
    Shape shape_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;

    Tensor(const Shape& shape) {
        shape_ = shape;

        size_t totalSize = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides_.push_back(totalSize);
            totalSize *= shape[i];
        }
        reverse(strides_.begin(), strides_.end());

        data_ = std::shared_ptr<U[]>(new U[totalSize]);
    }

    template <std::size_t... Is, typename... Args>
    Tensor(const Tensor& orig, std::index_sequence<Is...>, Args&&... args) {
        data_ = orig.data_;
        offset_ = orig.offset_;
        // Fold expression to process all arguments with overloads
        ((process_get_arg(orig, Is, args)), ...);
        
        // Copy remaining dimensions
        for (size_t i = sizeof...(Args); i < orig.strides_.size(); i++) {
            strides_.push_back(orig.strides_[i]);
            shape_.push_back(orig.shape_[i]);
        }
    }

    void process_get_arg(const Tensor& orig, std::size_t idx, int x) {
        if (idx >= orig.shape_.size()) {
            throw std::invalid_argument("Too many arguments provided for get.");
        }
        // Allow Python-style negative indexing
        if (x < 0) {
            x += orig.shape_[idx];
        }
        if (x >= (int)orig.shape_[idx] || x < 0) {
            throw std::invalid_argument("Index out of bounds.");
        }
        // Add to offset and don't add a range to axisIndices, reducing number of axes
        offset_ += orig.strides_[idx] * x;
    }

    void process_get_arg(const Tensor& orig, std::size_t idx, Range range) {
        if (idx >= orig.strides_.size()) {
            throw std::invalid_argument("Too many arguments provided for get.");
        }
        
        // Allow Python-style negative indexing
        if (range.start < 0) {
            range.start += orig.shape_[idx];
        }
        if (range.start < 0 || range.start >= (int)orig.shape_[idx]) {
            throw std::invalid_argument("Index out of bounds.");
        }

        // Length of -1 implies index to the end
        if (range.length == -1) {
            range.length = range.step > 0 ? (orig.shape_[idx] - range.start + range.step - 1) / range.step : (range.start / -range.step) + 1;
        }
        if (range.length <= 0) {
            throw std::invalid_argument("Negative length.");
        }
       
        shape_.push_back(range.length);
        strides_.push_back(orig.strides_[idx] * range.step);
        offset_ += orig.strides_[idx] * range.start;
    }

    // Write element wise op(a, b) into this object
    // Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
    template <typename Func>
    void apply_binary_helper(const Tensor& a, const Tensor& b, Func op, int index, int aIndex, int bIndex, int axis) {
        if (axis == (int)shape_.size()) {
            data_[index] = op(a.data_[aIndex], b.data_[bIndex]);
            return;
        }

        int aAxis = axis + a.shape_.size() - shape_.size();
        if (aAxis >= 0 && a.shape_[aAxis] == 1) {
            aAxis = -1;
        }
        
        int bAxis = axis + b.shape_.size() - shape_.size();
        if (bAxis >= 0 && b.shape_[bAxis] == 1) {
            bAxis = -1;
        }

        for (size_t i = 0; i < shape_[axis]; i++) {
            apply_binary_helper(
                a, b, op, 
                index + strides_[axis] * i, 
                aIndex + (aAxis >= 0 ? a.strides_[aAxis] * i : 0),
                bIndex + (bAxis >= 0 ? b.strides_[bAxis] * i : 0),
                axis + 1
            );
        }
    }

    // Write element wise op(a, b) into this object
    // Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
    template <typename Func>
    void apply_binary_inplace_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
        if (axis == (int)shape_.size()) {
            op(data_[index], other.data_[otherIndex]);
            return;
        }
        
        int otherAxis = axis + other.shape_.size() - shape_.size();
        if (otherAxis >= 0 && other.shape_[otherAxis] == 1) {
            otherAxis = -1;
        }

        for (size_t i = 0; i < shape_[axis]; i++) {
            apply_binary_inplace_helper(
                other, op, 
                index + strides_[axis] * i, 
                otherIndex + (otherAxis >= 0 ? other.strides_[otherAxis] * i : 0),
                axis + 1
            );
        }
    }

    // Apply op elementwise to other and copy into this. Guaranteed that this tensor and other are the same shape
    template <typename Func>
    void apply_unary_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
        if (axis == (int)shape_.size()) {
            data_[index] = op(other.data_[otherIndex]);
            return;
        }

        for (size_t i = 0; i < shape_[axis]; i++) {
            apply_unary_helper(other, op, index + strides_[axis] * i, otherIndex + other.strides_[axis] * i, axis + 1);
        }
    }

    // Apply elementwise
    template <typename Func>
    void apply_unary_inplace_helper(Func op, int index, int axis) {
        if (axis == (int)shape_.size()) {
            op(data_[index]);
            return;
        }

        for (size_t i = 0; i < shape_[axis]; i++) {
            apply_unary_inplace_helper(op, index + strides_[axis] * i, axis + 1);
        }
    }

    void matmul_helper(const Tensor& a, const Tensor& b, int index, int aIndex, int bIndex, int axis) {
        int aAxis = axis + a.shape_.size() - shape_.size();
        int bAxis = axis + b.shape_.size() - shape_.size();
        
        if (axis == (int)shape_.size() - 2) {
            // Actual matmul here
            for (size_t i = 0; i < shape_[axis]; i++) {
                for (size_t j = 0; j < shape_[axis + 1]; j++) {
                    for (size_t k = 0; k < a.shape_[aAxis + 1]; k++) {
                        data_[index + strides_[axis] * i + strides_[axis + 1] * j] += 
                            a.data_[aIndex + a.strides_[aAxis] * i + a.strides_[aAxis + 1] * k] * 
                            b.data_[bIndex + b.strides_[bAxis] * k + b.strides_[bAxis + 1] * j];
                    }
                }
            }
            return;
        }

        if (aAxis >= 0 && a.shape_[aAxis] == 1) {
            aAxis = -1;
        }

        if (bAxis >= 0 && b.shape_[bAxis] == 1) {
            bAxis = -1;
        }

        for (size_t i = 0; i < shape_[axis]; i++) {
            matmul_helper(
                a, b, 
                index + strides_[axis] * i, 
                aIndex + (aAxis >= 0 ? a.strides_[aAxis] * i : 0),
                bIndex + (bAxis >= 0 ? b.strides_[bAxis] * i : 0),
                axis + 1
            );
        }
    }

    void sum_helper(const Tensor& other, const std::vector<bool>& reduceAxis, int index, int otherIndex, int axis, int otherAxis) const {
        if (otherAxis == (int)other.shape_.size()) {
            data_[index] += other.data_[otherIndex];
            return;
        }        

        if (reduceAxis[otherAxis]) {
            for (size_t i = 0; i < other.shape_[otherAxis]; i++) {
                sum_helper(other, reduceAxis, index, otherIndex + other.strides_[otherAxis] * i, axis, otherAxis + 1);
            }
        }
        else {
            assert(shape_[axis] == other.shape_[otherAxis]);
            for (size_t i = 0; i < other.shape_[otherAxis]; i++) {
                sum_helper(other, reduceAxis, index + strides_[axis] * i, otherIndex + other.strides_[otherAxis] * i, axis + 1, otherAxis + 1);
            }
        }
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

} // namespace linalg

#endif // TENSOR_H


/*
Future optimizations:
- store backprop functions in arena instead of capturing lambdas to avoid heap allocations
- make data into an intrusive pointer to avoid an indirection


do proper tensor/tensorview with ownership so that we can be confident we're storing data
make tensors own data
*/