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
#include "tensor.hpp"

namespace linalg {

// Static private helpers
static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b);

int Range::operator[](int i) const {
    if (i >= length) {
        throw std::invalid_argument("Range index out of bounds.");
    }
    return start + i * step;
}

// Range composition
Range Range::operator[](const Range& other) const {
    if (other.start + (other.length - 1) * other.step >= length || other.start >= length || other.start + (other.length - 1) * other.step < 0 || other.start < 0) {
        throw std::invalid_argument("Range index out of bounds.");
    }
    return Range((*this)[other.start], other.length, step * other.step);
}

std::ostream& operator<<(std::ostream& os, const Range& t) {
    os << "Range(" << t.start << ", " << t.length << ", " << t.step << ")";
    return os;
}
 
// Constructor for 1D tensor
template <typename U>
Tensor<U>::Tensor(std::vector<U>&& values) : data(std::make_shared<std::vector<U>>()) {
    data->assign(values.begin(), values.end());
    shape = {values.size()};
    axisIndices = {Range(0, values.size(), 1)};
}

// Recursive constructor for arbitrary-dimensional tensors
template <typename U>
Tensor<U>::Tensor(std::vector<Tensor<U>>&& slices) : data(std::make_shared<std::vector<U>>()) {
    if (slices.size() == 0) {
        throw std::invalid_argument("Tensor cannot be empty");
    }

    // Determine the shape of the first slice
    shape = slices.begin()->shape;
    axisIndices = slices.begin()->axisIndices;

    // Check consistency of shapes across slices
    for (const auto& slice : slices) {
        if (slice.shape != shape) {
            throw std::invalid_argument("All slices must have the same shape");
        }
        data->insert(data->end(), slice.data->begin(), slice.data->end());
    }

    shape.insert(shape.begin(), slices.size());
    axisIndices.insert(axisIndices.begin(), Range(0, slices.size(), axisIndices.front().step * axisIndices.front().length));
}

//Tensor(U value) : data(std::make_shared<std::vector<U>>({value})) {}

template <typename U>
Tensor<U> Tensor<U>::zeros(const std::vector<size_t>& shape) {
    Tensor res;
    res.shape = shape;
    size_t totalSize = 1;
    
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        res.axisIndices.push_back(Range(0, shape[i], totalSize));
        totalSize *= shape[i];
    }
    
    reverse(res.axisIndices.begin(), res.axisIndices.end());
    res.data = std::make_shared<std::vector<U>>(std::vector<U>(totalSize, 0));
    return res;
}

template <typename U>
Tensor<U> Tensor<U>::normal(const std::vector<size_t>& shape, U mean, U std, std::optional<uint> seed) {
    Tensor res;
    res.shape = shape;
    size_t totalSize = 1;
    for (size_t i = shape.size() - 1; i >= 0; i--) {
        res.axisIndices.push_back(Range(0, shape[i], totalSize));
        totalSize *= shape[i];
    }
    reverse(res.axisIndices.begin(), res.axisIndices.end());
    res.data = std::make_shared<std::vector<U>>(std::vector<U>(totalSize, 0));

    std::mt19937 generator(seed ? *seed : std::random_device{}());
    std::normal_distribution<U> distribution(mean, std);
    for (auto& x : *res.data) {
        x = distribution(generator);
    }
    return res;
}

// Performs a deep copy that discards data not seen/accessed by the view
// Maybe try to remove recursion? incurs extra factor of #axes data copying
// Switch to using apply unary?
template <typename U>
Tensor<U> Tensor<U>::copy() const {
    if (shape.size() == 1) {
        std::vector<U> sliceCopies;
        for (size_t i = 0; i < shape[0]; i++) {
            sliceCopies.push_back((*this)[i]);
        }
        return Tensor(std::move(sliceCopies));
    }
    else {
        std::vector<Tensor> sliceCopies;
        for (size_t i = 0; i < shape[0]; i++) {
            sliceCopies.push_back((*this)[i].copy());
        }
        return Tensor(std::move(sliceCopies));
    }
}

// Variadic template access operator
template <typename U>
template <IntOrIterableOfInt... Args>
TensorProxy<U> Tensor<U>::operator[](Args... args) const {
    return TensorProxy<U>(*this, std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
}

// Apply some elementwise operation on two tensors with broadcasting
// Func should have some signature op(U, U) -> U
// Consider using template <auto op> with constexpr op instead of passing as parameter
template <typename U>
template <typename Func>
Tensor<U> Tensor<U>::apply_binary(const Tensor& other, Func op) const {
    std::vector<size_t> shape = broadcast_shape(this->shape, other.shape);

    if (shape.size() == 0) {
        throw std::invalid_argument("Broadcast failed.");
    }

    Tensor res = zeros(shape);
    res.apply_binary_helper((*this), other, op, 0, indexOffset, other.indexOffset, 0);
    return res;
}

// Apply some elementwise operation on two tensors with broadcasting, modifying the LHS tensor in place
// LHS tensor must be greater in all dimensions
// Func should have some signature op(U&, U) -> void
template <typename U>
template <typename Func>
Tensor<U>& Tensor<U>::apply_binary_inplace(const Tensor& other, Func op) {
    if (shape != broadcast_shape(this->shape, other.shape)) {
        throw std::invalid_argument("Broadcast failed.");
    }

    apply_binary_inplace_helper(other, op, indexOffset, other.indexOffset, 0);
    return *this;
}

// Apply some operation on some tensor, returning a new one
// Func should have signature op(U) -> void
template <typename U>
template <typename Func>
Tensor<U> Tensor<U>::apply_unary(Func op) const {
    Tensor res = zeros(shape);
    res.apply_unary_helper(*this, op, 0, indexOffset, 0);
    return res;
}

// Apply some operation on some tensor, modifying it in place
// Func should have signature op(U&) -> void
template <typename U>
template <typename Func>
Tensor<U>& Tensor<U>::apply_unary_inplace(Func op) {
    apply_unary_inplace_helper(op, indexOffset, 0);
    return *this;
}

template <typename U>
Tensor<U> Tensor<U>::operator-() const {
    return *this * (U)-1;
}

template <typename U>
Tensor<U> Tensor<U>::operator+(const Tensor& other) const {
    return apply_binary(other, [](U a, U b) { return a + b; });
}

template <typename U>
Tensor<U> Tensor<U>::operator-(const Tensor& other) const {
    return apply_binary(other, [](U a, U b) { return a - b; });
}

template <typename U>
Tensor<U> Tensor<U>::operator*(const Tensor& other) const {
    return apply_binary(other, [](U a, U b) { return a * b; });
}

template <typename U>
Tensor<U> Tensor<U>::operator/(const Tensor& other) const {
    return apply_binary(other, [](U a, U b) { return a / b; });
}

template <typename U>
Tensor<U> Tensor<U>::operator+(U other) const {
    return apply_unary([other](U a) { return a + other; });
}

template <typename U>
Tensor<U> Tensor<U>::operator-(U other) const {
    return apply_unary([other](U a) { return a - other; });
}

template <typename U>
Tensor<U> Tensor<U>::operator*(U other) const {
    return apply_unary([other](U a) { return a * other; });
}

template <typename U>
Tensor<U> Tensor<U>::operator/(U other) const {
    return apply_unary([other](U a) { return a / other; });
}

template <typename U>
Tensor<U>& Tensor<U>::operator+=(const Tensor& other) {
    apply_binary_inplace(other, [](U& a, U b) { a += b; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator-=(const Tensor& other) {
    apply_binary_inplace(other, [](U& a, U b) { a -= b; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator*=(const Tensor& other) {
    apply_binary_inplace(other, [](U& a, U b) { a *= b; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator/=(const Tensor& other) {
    apply_binary_inplace(other, [](U& a, U b) { a /= b; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator+=(U other) {
    apply_unary_inplace([other](U& a) { a += other; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator-=(U other) {
    apply_unary_inplace([other](U& a) { a -= other; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator*=(U other) {
    apply_unary_inplace([other](U& a) { a *= other; });
    return *this;
}

template <typename U>
Tensor<U>& Tensor<U>::operator/=(U other) {
    apply_unary_inplace([other](U& a) { a /= other; });
    return *this;
}

template <typename U>
Tensor<U> Tensor<U>::sum() {
    std::vector<int> axes(shape.size());
    for (size_t i = 0; i < shape.size(); i++) axes[i] = i;
    return sum(std::move(axes));
}

template <typename U>
Tensor<U> Tensor<U>::sum(int axis) {
    return sum(std::vector<int>{axis});
}

// Allows duplicates and ignores them
template <typename U>
Tensor<U> Tensor<U>::sum(std::vector<int>&& axes) {
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
template <typename U>
Tensor<U> Tensor<U>::matmul(const Tensor& a, const Tensor& b) {
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
        res.axisIndices.erase(res.axisIndices.end() - 2);
        res.shape.erase(res.shape.end() - 2);
    }
    if (b.shape.size() < 2) {
        // Remove 1 if b was reshaped
        assert(res.shape.back() == 1);
        res.axisIndices.pop_back();
        res.shape.pop_back();
    }
    return res;
}

template <typename U>
Tensor<U> Tensor<U>::T() const {
    Tensor res(*this);
    std::reverse(res.axisIndices.begin(), res.axisIndices.end());
    std::reverse(res.shape.begin(), res.shape.end());
    return res; // Should do NRVO
}

template <typename U>
Tensor<U> Tensor<U>::reshape(const std::vector<size_t>& newShape) const {
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

    std::vector<Range> newAxisIndices;

    // Logic is still a little sus but seems to work
    bool full = true;
    while (lNew < newShape.size()) {
        if (sizeCur < sizeNew && cur < shape.size()) {
            if (axisIndices[cur - 1].step != axisIndices[cur].step * axisIndices[cur].length || axisIndices[cur].start != 0)
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

            int stride = axisIndices[cur - 1].step;
            for (; lNew < rNew; lNew++) {
                sizeNew /= newShape[lNew];
                newAxisIndices.push_back(Range(0, newShape[lNew], sizeNew * stride));
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
    res.axisIndices = newAxisIndices;
    res.shape = newShape;

    return res;
}

//Cast Tensors with no dimension to scalar, implicit cast for cout/math
template <typename U>
Tensor<U>::operator U() const {
    if (shape.size() > 0) {
        throw std::invalid_argument("Can't cast tensor to scalar.");
    }
    return (*data)[indexOffset];
}

// Define << for printing using recursion
// template <typename U>
// std::ostream& operator<<(std::ostream& os, const Tensor<U>& t) {
//     if (t.shape.size() == 0) {
//         os << ((U)t);
//         return os;
//     }
//     os << "[";
//     for (size_t i = 0; i < t.shape[0]; i++) {
//         os << t[i];
//         if (i < t.shape[0] - 1)
//             os << ", ";
//     }
//     os << "]";

//     return os;
// }

//template std::ostream& operator<<(std::ostream& os, const Tensor<float>& t);

template <typename U>
void Tensor<U>::print() const {
    std::cout << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "): " << (*this) << std::endl;
}

template <typename U>
void Tensor<U>::print_axis_indices() {
    std::cout << "[";
    for (auto& range : axisIndices) {
        std::cout << range << ", ";
    }
    std::cout << indexOffset;
    // for (size_t i = 0; i < axisIndices.size(); i++) {
    //     std::cout << axisIndices[i];
    //     if (i < axisIndices.size() - 1)
    //         std::cout << ", ";
    // }
    std::cout << "]" << std::endl;
}

template <typename U>
void Tensor<U>::print_shape() const {
    for (auto dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}

// Write element wise op(a, b) into this object
// Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
template <typename U>
template <typename Func>
void Tensor<U>::apply_binary_helper(const Tensor& a, const Tensor& b, Func op, int index, int aIndex, int bIndex, int axis) {
    if (axis == (int)shape.size()) {
        (*data)[index] = op((*a.data)[aIndex], (*b.data)[bIndex]);
        return;
    }

    int aAxis = axis + a.shape.size() - shape.size();
    if (aAxis >= 0 && a.shape[aAxis] == 1) {
        aIndex += a.axisIndices[aAxis][0];
        aAxis = -1;
    }
    
    int bAxis = axis + b.shape.size() - shape.size();
    if (bAxis >= 0 && b.shape[bAxis] == 1) {
        bIndex += b.axisIndices[bAxis][0];
        bAxis = -1;
    }

    for (size_t i = 0; i < shape[axis]; i++) {
        apply_binary_helper(
            a, b, op, 
            index + axisIndices[axis][i], 
            aIndex + (aAxis >= 0 ? a.axisIndices[aAxis][i] : 0),
            bIndex + (bAxis >= 0 ? b.axisIndices[bAxis][i] : 0),
            axis + 1
        );
    }
}

// Write element wise op(a, b) into this object
// Do this to avoid instantiating many intermediate Tensors in our recursion, which causes a lot of data copying
template <typename U>
template <typename Func>
void Tensor<U>::apply_binary_inplace_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
    if (axis == (int)shape.size()) {
        op((*data)[index], (*other.data)[otherIndex]);
        return;
    }
    
    int otherAxis = axis + other.shape.size() - shape.size();
    if (otherAxis >= 0 && other.shape[otherAxis] == 1) {
        otherIndex += other.axisIndices[otherAxis][0];
        otherAxis = -1;
    }

    for (size_t i = 0; i < shape[axis]; i++) {
        apply_binary_inplace_helper(
            other, op, 
            index + axisIndices[axis][i], 
            otherIndex + (otherAxis >= 0 ? other.axisIndices[otherAxis][i] : 0),
            axis + 1
        );
    }
}

// Apply op elementwise to other and copy into this. Guaranteed that this tensor and other are the same shape
template <typename U>
template <typename Func>
void Tensor<U>::apply_unary_helper(const Tensor& other, Func op, int index, int otherIndex, int axis) {
    if (axis == (int)shape.size()) {
        (*data)[index] = op((*other.data)[otherIndex]);
        return;
    }

    for (size_t i = 0; i < shape[axis]; i++) {
        apply_unary_helper(other, op, index + axisIndices[axis][i], otherIndex + other.axisIndices[axis][i], axis + 1);
    }
}

// Apply elementwise
template <typename U>
template <typename Func>
void Tensor<U>::apply_unary_inplace_helper(Func op, int index, int axis) {
    if (axis == (int)shape.size()) {
        op((*data)[index]);
        return;
    }

    for (size_t i = 0; i < shape[axis]; i++) {
        apply_unary_inplace_helper(op, index + axisIndices[axis][i], axis + 1);
    }
}

template <typename U>
void Tensor<U>::matmul_helper(const Tensor& a, const Tensor& b, int index, int aIndex, int bIndex, int axis) {
    int aAxis = axis + a.shape.size() - shape.size();
    int bAxis = axis + b.shape.size() - shape.size();
    
    if (axis == (int)shape.size() - 2) {
        // Actual matmul here
        for (size_t i = 0; i < shape[axis]; i++) {
            for (size_t j = 0; j < shape[axis + 1]; j++) {
                for (size_t k = 0; k < a.shape[aAxis + 1]; k++) {
                    (*data)[index + axisIndices[axis][i] + axisIndices[axis + 1][j]] += 
                        (*a.data)[aIndex + a.axisIndices[aAxis][i] + a.axisIndices[aAxis + 1][k]] * 
                        (*b.data)[bIndex + b.axisIndices[bAxis][k] + b.axisIndices[bAxis + 1][j]];
                }
            }
        }
        return;
    }

    if (aAxis >= 0 && a.shape[aAxis] == 1) {
        aIndex += a.axisIndices[aAxis][0];
        aAxis = -1;
    }

    if (bAxis >= 0 && b.shape[bAxis] == 1) {
        bIndex += b.axisIndices[bAxis][0];
        bAxis = -1;
    }

    for (size_t i = 0; i < shape[axis]; i++) {
        matmul_helper(
            a, b, 
            index + axisIndices[axis][i], 
            aIndex + (aAxis >= 0 ? a.axisIndices[aAxis][i] : 0),
            bIndex + (bAxis >= 0 ? b.axisIndices[bAxis][i] : 0),
            axis + 1
        );
    }
}

template <typename U>
void Tensor<U>::sum_helper(const Tensor& other, const std::vector<bool>& reduceAxis, int index, int otherIndex, int axis, int otherAxis) {
    if (otherAxis == (int)other.shape.size()) {
        (*data)[index] += (*other.data)[otherIndex];
        return;
    }        

    if (reduceAxis[otherAxis]) {
        for (size_t i = 0; i < other.shape[otherAxis]; i++) {
            sum_helper(other, reduceAxis, index, otherIndex + other.axisIndices[otherAxis][i], axis, otherAxis + 1);
        }
    }
    else {
        assert(shape[axis] == other.shape[otherAxis]);
        for (size_t i = 0; i < other.shape[otherAxis]; i++) {
            sum_helper(other, reduceAxis, index + axisIndices[axis][i], otherIndex + other.axisIndices[otherAxis][i], axis + 1, otherAxis + 1);
        }
    }
}

// Returns empty vector on failure
std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
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
            return {};
        }
    }
    reverse(res.begin(), res.end());
    return res;
}

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

template <typename U>
template <std::size_t... Is, IntOrIterableOfInt... Args>
TensorProxy<U>::TensorProxy(const Tensor<U>& orig, std::index_sequence<Is...>, Args&&... args) {
    data = orig.data;
    indexOffset = orig.indexOffset;
    // Fold expression to process all arguments with overloads
    ((process_get_arg(orig, Is, args)), ...);
    
    // Copy remaining dimensions
    for (size_t i = sizeof...(Args); i < orig.axisIndices.size(); i++) {
        axisIndices.push_back(orig.axisIndices[i]);
        shape.push_back(orig.shape[i]);
    }
}

template <typename U>
TensorProxy<U>& TensorProxy<U>::operator=(U other) {
    Tensor<U>::apply_unary_inplace([other](U& a) { a = other; });
    return *this;
}

template <typename U>
TensorProxy<U>& TensorProxy<U>::operator=(const Tensor<U>& other) {
    Tensor<U>::apply_binary_inplace(other, [](U& a, U b) { a = b; });
    return *this;
}

template <typename U>
void TensorProxy<U>::process_get_arg(const Tensor<U>& orig, std::size_t idx, int x) {
    if (idx >= orig.axisIndices.size()) {
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
    indexOffset += orig.axisIndices[idx][x];
}

template <typename U>
void TensorProxy<U>::process_get_arg(const Tensor<U>& orig, std::size_t idx, const Range& range) {
    if (idx >= orig.axisIndices.size()) {
        throw std::invalid_argument("Too many arguments provided for get.");
    }
    Range copy(range);

    // Allow Python-style negative indexing
    if (copy.start < 0) {
        copy.start += orig.shape[idx];
    }
    if (copy.start < 0 || copy.start >= (int)orig.shape[idx]) {
        throw std::invalid_argument("Index out of bounds.");
    }

    // Length of 0 implies index to the end
    if (copy.length == 0) {
        copy.length = copy.step > 0 ? (orig.shape[idx] - copy.start + copy.step - 1) / copy.step : (copy.start / -copy.step) + 1;
    }
    if (copy.length <= 0) {
        throw std::invalid_argument("Negative length.");
    }

    // Range defines composition with operator[] which we use to compute the new range
    axisIndices.push_back(orig.axisIndices[idx][copy]);
    shape.push_back(axisIndices.back().length);

    indexOffset += axisIndices.back().start;
    axisIndices.back().start = 0;
}

template class Tensor<float>;
template class TensorProxy<float>;

} // namespace linalg