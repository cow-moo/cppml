#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include <memory>
#include <sstream>
#include "config.hpp"

namespace linalg {

class Strides {
public:
    Strides() : size_(0) {};
    Strides(const std::vector<size_t>& init) : size_(init.size()) {
        std::copy(init.begin(), init.end(), dims_.begin());
    }

    Strides(const std::initializer_list<size_t>& init) : size_(init.size()) {
        std::copy(init.begin(), init.end(), dims_.begin());
    }

    using iterator = typename std::array<size_t, config::MAX_DIMS>::iterator;

    auto front() const { return dims_.front(); }
    auto back() const { return dims_[size_ - 1]; }

    auto begin() { return dims_.begin(); }
    auto end() { return dims_.begin() + size_; }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.begin() + size_; }

    void push_back(size_t val) {
        assert(size_ < config::MAX_DIMS);
        dims_[size_] = val;
        ++size_;
    }

    void pop_back() { --size_; }

    auto insert(iterator pos, size_t val) {
        assert(size_ < config::MAX_DIMS);
        size_t idx = pos - dims_.begin();
        for (size_t i = size_; i > idx; --i)
            dims_[i] = dims_[i - 1];
        dims_[idx] = val;
        ++size_;
        return pos;
    }

    auto erase(iterator pos) {
        assert(size_ > 0);
        size_t idx = pos - dims_.begin();
        for (size_t i = idx; i < size_ - 1; ++i) {
            dims_[i] = dims_[i + 1];
        }
        --size_;
        return pos;
    }

    bool operator==(const Strides& other) const {
        if (size_ != other.size_)
            return false;
        for (size_t i = 0; i < size_; ++i) {
            if (dims_[i] != other.dims_[i])
                return false;
        }
        return true;
    }

    bool operator!=(const Strides& other) const {
        return !(*this == other);
    }

    size_t& operator[](size_t i) { return dims_[i]; }
    const size_t& operator[](size_t i) const { return dims_[i]; }

    size_t& operator[](int i) {
        if (i < 0) i += size_;
        if (i < 0 || i >= (int)size_) {
            throw std::invalid_argument("Axis index out of bounds.");
        }
        return dims_[i];
    }

    const size_t& operator[](int i) const {
        if (i < 0) i += size_;
        if (i < 0 || i >= (int)size_) {
            throw std::invalid_argument("Axis index out of bounds.");
        }
        return dims_[i];
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    friend std::ostream& operator<<(std::ostream& os, const Strides& t) {
        os << "(";
        for (size_t i = 0; i < t.size(); i++) {
            os << t[i];
            if (i < t.size() - 1)
                os << ", ";
        }
        os << ")";
        return os;
    }

private:
    std::array<size_t, config::MAX_DIMS> dims_;
    size_t size_;
};

class Shape {
public:
    Shape() : size_(0) {};
    Shape(const std::vector<size_t>& init) : size_(init.size()) {
        std::copy(init.begin(), init.end(), dims_.begin());
    }

    Shape(const std::initializer_list<size_t>& init) : size_(init.size()) {
        std::copy(init.begin(), init.end(), dims_.begin());
    }

    using iterator = typename std::array<size_t, config::MAX_DIMS>::iterator;
    
    // Forward vector methods
    auto front() const { return dims_.front(); }
    auto back() const { return dims_[size_ - 1]; }

    auto begin() { return dims_.begin(); }
    auto end() { return dims_.begin() + size_; }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.begin() + size_; }

    void push_back(size_t val) {
        assert(size_ < config::MAX_DIMS);
        dims_[size_] = val;
        ++size_;
    }

    void pop_back() { --size_; }

    auto insert(iterator pos, size_t val) {
        assert(size_ < config::MAX_DIMS);
        size_t idx = pos - dims_.begin();
        for (size_t i = size_; i > idx; --i)
            dims_[i] = dims_[i - 1];
        dims_[idx] = val;
        ++size_;
        return pos;
    }

    auto erase(iterator pos) {
        assert(size_ > 0);
        size_t idx = pos - dims_.begin();
        for (size_t i = idx; i < size_ - 1; ++i) {
            dims_[i] = dims_[i + 1];
        }
        --size_;
        return pos;
    }

    bool operator==(const Shape& other) const {
        if (size_ != other.size_)
            return false;
        for (size_t i = 0; i < size_; ++i) {
            if (dims_[i] != other.dims_[i])
                return false;
        }
        return true;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    size_t& operator[](size_t i) { return dims_[i]; }
    const size_t& operator[](size_t i) const { return dims_[i]; }

    size_t& operator[](int i) {
        if (i < 0) i += size_;
        if (i < 0 || i >= (int)size_) {
            throw std::invalid_argument("Axis index out of bounds.");
        }
        return dims_[i];
    }

    const size_t& operator[](int i) const {
        if (i < 0) i += size_;
        if (i < 0 || i >= (int)size_) {
            throw std::invalid_argument("Axis index out of bounds.");
        }
        return dims_[i];
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    size_t numel() const {
        size_t res = 1;
        for (size_t i = 0; i < size_; ++i)
            res *= dims_[i];
        return res;
    }

    static Shape broadcast(const Shape& a, const Shape& b) {
        Shape res;
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
                std::cout << a << " " << b << std::endl;
                throw std::invalid_argument("Broadcast failed.");
            }
        }
        std::reverse(res.begin(), res.end());
        return Shape{res};
    }

    static auto broadcast(const Shape& a, const Strides& aStrides, const Shape& b, const Strides& bStrides) {
        Shape res;
        Strides aRes, bRes;
        for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
            if (i >= a.size()) {
                res.push_back(b[b.size() - 1 - i]);
                aRes.push_back(0);
                bRes.push_back(bStrides[b.size() - 1 - i]);
            }
            else if (i >= b.size()) {
                res.push_back(a[a.size() - 1 - i]);
                aRes.push_back(aStrides[a.size() - 1 - i]);
                bRes.push_back(0);
            }
            else if (a[a.size() - 1 - i] == 1) {
                res.push_back(b[b.size() - 1 - i]);
                aRes.push_back(0);
                bRes.push_back(bStrides[b.size() - 1 - i]);
            }
            else if (b[b.size() - 1 - i] == 1) {
                res.push_back(a[a.size() - 1 - i]);
                aRes.push_back(aStrides[a.size() - 1 - i]);
                bRes.push_back(0);
            }
            else if (a[a.size() - 1 - i] == b[b.size() - 1 - i]) {
                res.push_back(a[a.size() - 1 - i]);
                aRes.push_back(aStrides[a.size() - 1 - i]);
                bRes.push_back(bStrides[b.size() - 1 - i]);
            }
            else {
                std::cout << a << " " << b << std::endl;
                throw std::invalid_argument("Broadcast failed.");
            }
        }
        std::reverse(res.begin(), res.end());
        std::reverse(aRes.begin(), aRes.end());
        std::reverse(bRes.begin(), bRes.end());

        assert(res.size() == aRes.size() && res.size() == bRes.size());
        return std::tuple{res, aRes, bRes};
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& t) {
        os << "(";
        for (size_t i = 0; i < t.size(); i++) {
            os << t[i];
            if (i < t.size() - 1)
                os << ", ";
        }
        os << ")";
        return os;
    }

private:
    std::array<size_t, config::MAX_DIMS> dims_;
    size_t size_;
};

}

#endif // SHAPE_H