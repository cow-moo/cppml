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
    struct NestedInitializer;

    Tensor(const NestedInitializer& initializer, backend::BackendType type = backend::current_backend_type) : Tensor(initializer.shape, type) {
        data_->write_flat(initializer.flatData);
    }

    Tensor(const Shape& shape, backend::BackendType type = backend::current_backend_type) 
        : data_(shape.numel(), type), shape_(shape) {
        size_t totalSize = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides_.push_back(totalSize);
            totalSize *= shape[i];
        }
        reverse(strides_.begin(), strides_.end());
    }

    // Needed to resolve ambiguity between NestedInitializer and Shape constructors
    Tensor(const std::initializer_list<U>& list, backend::BackendType type = backend::current_backend_type)
        : Tensor(NestedInitializer(list), type) {}

    Tensor(const Tensor& other)
        : data_(other.data_), shape_(other.shape_), strides_(other.strides_), offset_(other.offset_) {}

    // Elementwise assignment
    Tensor& operator=(U other) {
        return apply_binary_inplace(other, backend::BinOp::Pass);
    }

    // Elementwise assignment
    Tensor& operator=(const Tensor& other) {
        return apply_binary_inplace(other, backend::BinOp::Pass);
    }

    // Rebinding assignment
    Tensor& assign(const Tensor& other) {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        offset_ = other.offset_;
        return *this;
    }

    static Tensor zeros(const Shape& shape, backend::BackendType type = backend::current_backend_type) {
        Tensor res(shape, type);

        std::vector<U> flat(shape.numel(), 0);
        res.data_->write_flat(flat);

        return res;
    }

    static Tensor normal(const Shape& shape, U mean = 0, U std = 1, std::optional<uint> seed = std::nullopt, backend::BackendType type = backend::current_backend_type) {
        Tensor res(shape, type);

        std::mt19937 generator(seed ? *seed : std::random_device{}());
        std::normal_distribution<U> distribution(mean, std);

        std::vector<U> flat;
        flat.reserve(shape.numel());

        for (size_t i = 0; i < shape.numel(); i++) {
            flat.push_back(distribution(generator));
        }

        res.data_->write_flat(flat);

        return res;
    }

    static Tensor normal(const Shape& shape, backend::BackendType type) {
        return normal(shape, 0, 1, std::nullopt, type);
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
        return at(std::forward<Args>(args)...);
    }

    // Needed due to clang issues with perfect forwarding into operator[]
    template <typename... Args>
    Tensor at(Args&&... args) const {
        return Tensor(*this, std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
    }

    template <typename V>
    friend Tensor<V> operator+(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<V> operator-(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<V> operator*(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<V> operator/(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<V> operator-(V a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<V> operator/(V a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<bool> operator==(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<bool> operator<(const Tensor<V>& a, const Tensor<V>& b);

    template <typename V>
    friend Tensor<bool> operator<=(const Tensor<V>& a, const Tensor<V>& b);

    Tensor<bool> operator<(U other) const {
        return apply_binary<bool>(other, backend::BinOp::Lt);
    }

    Tensor<bool> operator>(U other) const {
        return apply_binary<bool>(other, backend::BinOp::Gt);
    }

    Tensor<bool> operator<=(U other) const {
        return apply_binary<bool>(other, backend::BinOp::Lte);
    }

    Tensor<bool> operator>=(U other) const {
        return apply_binary<bool>(other, backend::BinOp::Gte);
    }

    Tensor operator-() const {
        return neg();
    }

    Tensor operator+(U other) const {
        return apply_binary(other, backend::BinOp::Add);
    }

    Tensor operator-(U other) const {
        return apply_binary(other, backend::BinOp::Sub);
    }

    Tensor operator*(U other) const {
        return apply_binary(other, backend::BinOp::Mul);
    }

    Tensor operator/(U other) const {
        return apply_binary(other, backend::BinOp::Div);
    }

    Tensor& operator+=(const Tensor& other) {
        return apply_binary_inplace(other, backend::BinOp::Add);
    }

    Tensor& operator-=(const Tensor& other) {
        return apply_binary_inplace(other, backend::BinOp::Sub);
    }

    Tensor& operator*=(const Tensor& other) {
        return apply_binary_inplace(other, backend::BinOp::Mul);
    }

    Tensor& operator/=(const Tensor& other) {
        return apply_binary_inplace(other, backend::BinOp::Div);
    }

    Tensor& operator+=(U other) {
        return apply_binary_inplace(other, backend::BinOp::Add);
    }

    Tensor& operator-=(U other) {
        return apply_binary_inplace(other, backend::BinOp::Sub);
    }

    Tensor& operator*=(U other) {
        return apply_binary_inplace(other, backend::BinOp::Mul);
    }

    Tensor& operator/=(U other) {
        return apply_binary_inplace(other, backend::BinOp::Div);
    }

    Tensor exp() const {
        return apply_unary(backend::UnOp::Exp);
    }

    Tensor& exp_() {
        return apply_unary_inplace(backend::UnOp::Exp);
    }

    Tensor log() const {
        return apply_unary(backend::UnOp::Log);
    }

    Tensor& log_() {
        return apply_unary_inplace(backend::UnOp::Log);
    }

    Tensor neg() const {
        return apply_unary(backend::UnOp::Neg);
    }

    Tensor& neg_() {
        return apply_unary_inplace(backend::UnOp::Neg);
    }

    Tensor relu() const {
        return ((*this) > 0.0f).template astype<U>() * (*this);
    }

    Tensor<bool> operator==(U other) const {
        return apply_binary<bool>(other, backend::BinOp::Eq);
    }
    
    // Allows duplicates and ignores them
    Tensor sum(const std::vector<int>& axes, bool keepDims = false) const {
        return reduce(axes, 0, backend::BinOp::Add, keepDims);
    }

    Tensor sum() const {
        std::vector<int> axes(shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) axes[i] = i;
        return sum(axes);
    }

    Tensor max(const std::vector<int>& axes, bool keepDims = false) const {
        return reduce(axes, std::numeric_limits<U>::lowest(), 
                      backend::BinOp::Max, keepDims);
    }

    Tensor min(const std::vector<int>& axes, bool keepDims = false) const {
        return reduce(axes, std::numeric_limits<U>::max(), 
                      backend::BinOp::Min, keepDims);
    }

    Tensor<size_t> argmax(int axis = -1, bool keepDim = false) const {
        return arg_reduce(axis, backend::ArgRedOp::Max, keepDim);
    }

    Tensor<size_t> argmin(int axis = -1, bool keepDim = false) const {
        return arg_reduce(axis, backend::ArgRedOp::Min, keepDim);
    }

    // Broadcasts first n-2 dimensions
    // 1D tensors have 1 appended (column vector)
    friend Tensor matmul(const Tensor& a, const Tensor& b) {
        assert(a.shape().size() > 0 && b.shape().size() > 0);
        assert(a.data_->backend_type() == b.data_->backend_type());

        Shape aShape(a.shape()), bShape(b.shape());
        Strides aStrides(a.strides_), bStrides(b.strides_);

        if (aShape.size() == 1) {
            aShape.push_back(1);
            aStrides.push_back(0);
        }
        if (bShape.size() == 1) {
            bShape.push_back(1);
            bStrides.push_back(0);
        }

        if (aShape[-1] != bShape[-2]) {
            std::cout << a.shape_ << " " << b.shape_ << std::endl;
            throw std::invalid_argument("Invalid shape for matmul.");
        }

        Shape aOuterShape(aShape);
        aOuterShape.pop_back(); aOuterShape.pop_back();
        Shape bOuterShape(bShape);
        bOuterShape.pop_back(); bOuterShape.pop_back();
        Strides aOuterStrides(aStrides);
        aOuterStrides.pop_back(); aOuterStrides.pop_back();
        Strides bOuterStrides(bStrides);
        bOuterStrides.pop_back(); bOuterStrides.pop_back();

        Shape rShape;

        std::tie(rShape, aOuterStrides, bOuterStrides) = Shape::broadcast(aOuterShape, aOuterStrides, bOuterShape, bOuterStrides);

        rShape.push_back(aShape[-2]); rShape.push_back(bShape[-1]);
        aOuterStrides.push_back(aStrides[-2]); aOuterStrides.push_back(aStrides[-1]);
        bOuterStrides.push_back(bStrides[-2]); bOuterStrides.push_back(bStrides[-1]);

        Tensor res(rShape, a.data_->backend_type());

        res.data_->matmul(rShape, res.strides_, res.offset_,
                          a.data_.get(), aOuterStrides, a.offset_,
                          b.data_.get(), bOuterStrides, b.offset_,
                          aShape[-1]);

        if (b.shape_.size() == 1)
            return res.squeeze({-1});
        else
            return res;
    }

    Tensor softmax(int axis = -1) const {
        Tensor scratch = *this - max({axis}, true);
        scratch.exp_();
        Tensor sum = scratch.sum({axis}, true);
        return scratch / sum;
    }

    Tensor log_softmax(int axis = -1) const {
        Tensor normalized = (*this) - max({axis}, true);
        Tensor sum = normalized.exp().sum({axis}, true);
        return normalized - sum.log_();
    }

    Tensor T() const {
        assert(shape_.size() > 0);
        if (shape_.size() < 2) {
            return reshape({1, shape_[0]});
        }

        Tensor res(*this);
        std::swap(res.shape_[-1], res.shape_[-2]);
        std::swap(res.strides_[-1], res.strides_[-2]);
        return res;
    }

    Tensor reshape(const Shape& newShape) const {
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

        if (shape_.size() == 0) {
            Tensor res(*this);
            for (size_t i = 0; i < newShape.size(); i++) {
                res.strides_.push_back(1);
            }
            res.shape_ = newShape;
            return res;
        }

        size_t cur = 1;
        size_t sizeCur = shape_[0];
        size_t lNew = 0, rNew = 0;
        size_t sizeNew = 1;

        Strides newStrides;

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

    Tensor unsqueeze(int axis) const {
        if (axis < 0) axis += shape_.size();
        if (axis < 0 || axis > (int)shape_.size()) {
            throw std::invalid_argument("Axis index out of bounds.");
        }

        Shape newShape = shape_;
        newShape.insert(newShape.begin() + axis, 1);

        return reshape(newShape);
    }

    Tensor squeeze(const std::vector<int>& axes) const {
        std::vector<bool> reduceAxis(shape_.size(), false);

        for (auto x : axes) {
            if (x < 0) x += shape_.size();
            if (x < 0 || x >= (int)shape_.size()) {
                throw std::invalid_argument("Axis index out of bounds.");
            }

            reduceAxis[x] = true;
            if (shape_[x] != 1) {
                throw std::invalid_argument("Squeeze failed.");
            }
        }

        Shape newShape;
        for (size_t i = 0; i < shape_.size(); i++) {
            if (!reduceAxis[i])
                newShape.push_back(shape_[i]);
        }

        return reshape(newShape);
    }

    size_t numel() const {
        return shape_.numel();
    }

    template <typename R>
    Tensor<R> astype() const {
        return apply_unary<R>(backend::UnOp::Pass);
    }

    Tensor to(backend::BackendType type) const {
        Tensor res(shape_, type);
        res.data_->write_flat(data_->read_strided(shape_, strides_, offset_));
        return res;
    }

    backend::BackendType backend_type() const {
        return data_->backend_type();
    }

    // Cast Tensors with no dimension to scalar, implicit cast for cout/math
    operator U() const {
        if (shape_.size() > 0) {
            throw std::invalid_argument("Can't cast tensor to scalar.");
        }
        return data_->at(offset_);
    }

    // Define << for printing using recursion
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (t.shape_.size() == 0) {
            os << ((U)t);
            return os;
        }
        auto strided = t.data_->read_strided(t.shape_, t.strides_, t.offset_);

        for (size_t i = 0; i < t.shape_.size(); i++)
            os << "[";
        for (size_t i = 0; i < strided.size(); i++) {
            if (i != 0) {
                size_t depth = 0, ii = i;
                while (depth < t.shape_.size()) {
                    if (ii % t.shape_[t.shape_.size() - 1 - depth] != 0)
                        break;
                    ii /= t.shape_[t.shape_.size() - 1 - depth];
                    depth++;
                }
                for (size_t j = 0; j < depth; j++)
                    os << "]";
                os << ", ";
                for (size_t j = 0; j < depth; j++)
                    os << "[";
            }
            os << strided[i];
        }
        for (size_t i = 0; i < t.shape_.size(); i++)
            os << "]";

        return os;
    }

    void print() const {
        std::cout << data_->backend_type() << " " << shape_;
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

        return sum(broadcastedAxes);
    }

    const Shape& shape() const {
        return shape_;
    }

    // Could probably rewrite to avoid intermediate allocations but probably not worth it
    struct NestedInitializer {
        std::vector<U> flatData;
        Shape shape;

        NestedInitializer(const std::initializer_list<U>& slice) 
            : flatData(slice), shape({slice.size()}) {}

        NestedInitializer(const std::initializer_list<NestedInitializer>& slices) {
            shape = slices.begin()->shape;
            for (auto& slice : slices) {
                if (shape != slice.shape) {
                    throw std::invalid_argument("All slices must have the same shape.");
                }
                flatData.insert(flatData.end(), slice.flatData.begin(), slice.flatData.end());
            }
            shape.insert(shape.begin(), slices.size());
        }

        NestedInitializer(const std::vector<U>& flat) 
            : flatData(flat), shape({flat.size()}) {}
    };

private:
    // Should be fixed on construction
    backend::SharedBuffer<U> data_;
    Shape shape_;
    Strides strides_;
    size_t offset_ = 0;

    template <typename>
    friend class Tensor;

    template <std::size_t... Is, typename... Args>
    Tensor(const Tensor& orig, std::index_sequence<Is...>, Args&&... args) : data_(orig.data_) {
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

    template <typename R = U, typename V = U>
    Tensor<R> apply_binary(const Tensor<V>& other, backend::BinOp op) const {
        assert(data_->backend_type() == other.data_->backend_type());
        auto [shape, strides, otherStrides] = Shape::broadcast(shape_, strides_, other.shape_, other.strides_);

        Tensor<R> res(shape, data_->backend_type());

        // We can const_cast because we know res != this and other
        res.data_->apply_binary(shape, res.strides_, res.offset_,
                                const_cast<backend::DeviceBuffer<U>*>(data_.get()), strides, offset_,
                                const_cast<backend::DeviceBuffer<V>*>(other.data_.get()), otherStrides, other.offset_,
                                op);

        return res;
    }

    template <typename R = U, typename V = U>
    Tensor<R> apply_binary(V other, backend::BinOp op) const {
        Tensor<R> res(shape_, data_->backend_type());
        res.data_->apply_binary(shape_, res.strides_, res.offset_,
                                const_cast<backend::DeviceBuffer<U>*>(data_.get()), strides_, offset_,
                                other, op);
        return res;
    }

    // Apply some elementwise operation on two tensors with broadcasting, modifying the LHS tensor in place
    // LHS tensor must be greater in all dimensions
    // Func should have some signature op(U&, U) -> void
    template <typename V = U>
    Tensor& apply_binary_inplace(const Tensor<V>& other, backend::BinOp op) {
        auto [shape, _, otherStrides] = Shape::broadcast(shape_, strides_, other.shape_, other.strides_);
        
        if (shape_ != shape) {
            std::cout << this->shape_ << " " << other.shape_ << std::endl;
            throw std::invalid_argument("Broadcast failed.");
        }

        data_->apply_binary(shape_, strides_, offset_,
                            data_.get(), strides_, offset_,
                            const_cast<backend::DeviceBuffer<U>*>(other.data_.get()), otherStrides, other.offset_,
                            op);

        return *this;
    }

    template <typename V>
    Tensor& apply_binary_inplace(V other, backend::BinOp op) {
        data_->apply_binary(shape_, strides_, offset_,
                            data_.get(), strides_, offset_,
                            other, op);

        return *this;
    }

    // Apply some operation on some tensor, returning a new one
    // Func should have signature op(U) -> R
    template <typename R = U>
    Tensor<R> apply_unary(backend::UnOp op) const {
        Tensor<R> res = Tensor<R>(shape_, data_->backend_type());
        // We can const_cast because we know res != this
        res.data_->apply_unary(shape_, res.strides_, res.offset_,
                               const_cast<backend::DeviceBuffer<U>*>(data_.get()), strides_, offset_, 
                               op);
        return res;
    }

    // Apply some operation on some tensor, modifying it in place
    // Func should have signature op(U&) -> void
    Tensor& apply_unary_inplace(backend::UnOp op) {
        data_->apply_unary(shape_, strides_, offset_,
                           data_.get(), strides_, offset_, 
                           op);
        return *this;
    }

    // Allows duplicate axes and ignores them
    Tensor reduce(const std::vector<int>& axes, U identity, backend::BinOp op, bool keepDims) const {
        if (axes.size() == 0)
            return copy();

        std::vector<bool> reduceAxis(shape_.size(), false);

        for (auto x : axes) {
            if (x < 0) x += shape_.size();
            if (x < 0 || x >= (int)shape_.size()) {
                throw new std::invalid_argument("Bad axis for reduce");
            }

            reduceAxis[x] = true;
        }

        Shape rShape, reduceShape;
        Strides otherStrides;
        for (size_t i = 0; i < shape_.size(); i++) {
            if (!reduceAxis[i]) {
                rShape.push_back(shape_[i]);
                otherStrides.push_back(strides_[i]);
            }
        }
        for (size_t i = 0; i < shape_.size(); i++) {
            if (reduceAxis[i]) {
                reduceShape.push_back(shape_[i]);
                otherStrides.push_back(strides_[i]);
            }
        }

        Tensor res(rShape, data_->backend_type());

        res.data_->reduce(rShape, res.strides_, res.offset_,
                          data_.get(), otherStrides, offset_,
                          reduceShape, identity, op);

        if (keepDims) {
            Shape newShape(shape_);
            for (auto x : axes)
                newShape[x] = 1;
            return res.reshape(newShape);
        }

        return res;

        // Shape newShape(shape_);

        // for (auto x : axes) {
        //     newShape[x] = 1;
        // }
        
        // Tensor res(newShape, data_->backend_type());
        // res = identity;

        // Strides strides(res.strides_);
        // for (auto x : axes) {
        //     if (x < 0) x += shape_.size();
        //     strides[x] = 0;
        // }

        // res.data_->apply_binary(shape_, strides, res.offset_,
        //                         res.data_.get(), strides, res.offset_,
        //                         const_cast<backend::DeviceBuffer<U>*>(data_.get()), strides_, offset_,
        //                         op);

        // if (keepDims)
        //     return res;
        // else
        //     return res.squeeze(axes);
    }

    Tensor<size_t> arg_reduce(int axis, backend::ArgRedOp op, bool keepDim) const {
        assert(shape_.size() > 0);

        if (axis < 0) axis += shape_.size();

        Shape shape(shape_);
        Strides strides(strides_);
        std::swap(shape[-1], shape[axis]);
        std::swap(strides[-1], strides[axis]);

        Shape newShape(shape);
        newShape.pop_back();
        Tensor<size_t> res(newShape, data_->backend_type());
        res.data_->arg_reduce(newShape, res.strides_, res.offset_,
                              data_.get(), strides, offset_,
                              shape.back(), op);

        if (keepDim)
            return res.unsqueeze(axis);
        else
            return res;
    }
};

template <typename U>
Tensor<U> operator+(const Tensor<U>& a, const Tensor<U>& b) {
    return a.apply_binary(b, backend::BinOp::Add);
}

template <typename U>
Tensor<U> operator-(const Tensor<U>& a, const Tensor<U>& b) {
    return a.apply_binary(b, backend::BinOp::Sub);
}

template <typename U>
Tensor<U> operator*(const Tensor<U>& a, const Tensor<U>& b) {
    return a.apply_binary(b, backend::BinOp::Mul);
}

template <typename U>
Tensor<U> operator/(const Tensor<U>& a, const Tensor<U>& b) {
    return a.apply_binary(b, backend::BinOp::Div);
}

template <typename U>
Tensor<U> operator+(U a, const Tensor<U>& b) {
    return b + a;
}

template <typename U>
Tensor<U> operator-(U a, const Tensor<U>& b) {
    return b.apply_binary(a, backend::BinOp::SubBy);
}

template <typename U>
Tensor<U> operator*(U a, const Tensor<U>& b) {
    return b * a;
}

template <typename U>
Tensor<U> operator/(U a, const Tensor<U>& b) {
    return b.apply_binary(a, backend::BinOp::DivBy);
}

template <typename U>
Tensor<bool> operator==(const Tensor<U>& a, const Tensor<U>& b) {
    return a.template apply_binary<bool>(b, backend::BinOp::Eq);
}

template <typename U>
Tensor<bool> operator==(U a, const Tensor<U>& b) {
    return b == a;
}

template <typename U>
Tensor<bool> operator<(const Tensor<U>& a, const Tensor<U>& b) {
    return a.template apply_binary<bool>(b, backend::BinOp::Lt);
}

template <typename U>
Tensor<bool> operator>(const Tensor<U>& a, const Tensor<U>& b) {
    return b < a;
}

template <typename U>
Tensor<bool> operator<=(const Tensor<U>& a, const Tensor<U>& b) {
    return a.template apply_binary<bool>(b, backend::BinOp::Lte);
}

template <typename U>
Tensor<bool> operator>=(const Tensor<U>& a, const Tensor<U>& b) {
    return b <= a;
}

template <typename U>
Tensor<bool> operator<(U a, const Tensor<U>& b) {
    return b > a;
}

template <typename U>
Tensor<bool> operator>(U a, const Tensor<U>& b) {
    return b < a;
}

template <typename U>
Tensor<bool> operator<=(U a, const Tensor<U>& b) {
    return b >= a;
}

template <typename U>
Tensor<bool> operator>=(U a, const Tensor<U>& b) {
    return b <= a;
}

template <typename U>
Tensor<U> exp(const Tensor<U>& t) {
    return t.exp();
}

template <typename U>
Tensor<U> log(const Tensor<U>& t) {
    return t.log();
}

template <typename U>
Tensor<U> sum(const Tensor<U>& t) {
    return t.sum();
}

} // namespace linalg

#endif // TENSOR_H


/*
Future optimizations:
- store backprop functions in arena instead of capturing lambdas to avoid heap allocations
- make data into an intrusive pointer to avoid an indirection
- SBO for shapes and strides
- consider coalescing axes when doing elementwise operations

do proper tensor/tensorview with ownership so that we can be confident we're storing data
make tensors own data
*/