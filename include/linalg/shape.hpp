#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include <array>
#include <memory>
#include <sstream>
#include <cassert>
#include <iostream>
#include "config.hpp"

namespace linalg {

class Shape;

class Strides {
public:
    Strides() : size_(0) {};
    Strides(const std::vector<size_t>& init);
    Strides(const std::initializer_list<size_t>& init);
    Strides(const Shape& shape);

    using iterator = typename std::array<size_t, config::MAX_DIMS>::iterator;

    size_t* data() { return dims_.data(); }
    const size_t* data() const { return dims_.data(); }

    std::array<size_t, config::MAX_DIMS> array() { return dims_; }
    const std::array<size_t, config::MAX_DIMS> array() const { return dims_; }

    auto front() const { return dims_.front(); }
    auto back() const { return dims_[size_ - 1]; }

    auto begin() { return dims_.begin(); }
    auto end() { return dims_.begin() + size_; }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.begin() + size_; }

    void push_back(size_t val);

    void pop_back() { --size_; }

    iterator insert(iterator pos, size_t val);

    iterator erase(iterator pos);

    bool operator==(const Strides& other) const;

    bool operator!=(const Strides& other) const { return !(*this == other); }

    size_t& operator[](size_t i) { return dims_[i]; }
    const size_t& operator[](size_t i) const { return dims_[i]; }

    size_t& operator[](int i);
    const size_t& operator[](int i) const;

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    friend std::ostream& operator<<(std::ostream& os, const Strides& t);

private:
    std::array<size_t, config::MAX_DIMS> dims_;
    size_t size_;
};

class Shape {
public:
    Shape() : size_(0) {};
    Shape(const std::vector<size_t>& init);

    Shape(const std::initializer_list<size_t>& init);

    using iterator = typename std::array<size_t, config::MAX_DIMS>::iterator;
    
    size_t* data() { return dims_.data(); }
    const size_t* data() const { return dims_.data(); }

    std::array<size_t, config::MAX_DIMS> array() { return dims_; }
    const std::array<size_t, config::MAX_DIMS> array() const { return dims_; }

    auto front() const { return dims_.front(); }
    auto back() const { return dims_[size_ - 1]; }

    auto begin() { return dims_.begin(); }
    auto end() { return dims_.begin() + size_; }
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.begin() + size_; }

    void push_back(size_t val);

    void pop_back() { --size_; }

    iterator insert(iterator pos, size_t val);

    iterator erase(iterator pos);

    bool operator==(const Shape& other) const;

    bool operator!=(const Shape& other) const { return !(*this == other); }

    size_t& operator[](size_t i) { return dims_[i]; }
    const size_t& operator[](size_t i) const { return dims_[i]; }

    size_t& operator[](int i);
    const size_t& operator[](int i) const;

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    size_t numel() const;

    static Shape broadcast(const Shape& a, const Shape& b);

    static std::tuple<Shape, Strides, Strides> broadcast(
        const Shape& a, const Strides& aStrides, 
        const Shape& b, const Strides& bStrides);

    friend std::ostream& operator<<(std::ostream& os, const Shape& t);

private:
    std::array<size_t, config::MAX_DIMS> dims_;
    size_t size_;
};

}

#endif // SHAPE_H