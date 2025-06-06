#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include <memory>
#include <sstream>

namespace linalg {

// TODO: reimplement with SBO

using Strides = std::vector<size_t>;

struct Shape {
    std::vector<size_t> dims;

    Shape() = default;
    Shape(std::vector<size_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<size_t> init) : dims(init) {}

    operator std::vector<size_t>&() { return dims; }
    operator const std::vector<size_t>&() const { return dims; }

    // Forward vector methods
    auto front() const { return dims.front(); }
    auto back() const { return dims.back(); }

    auto begin() { return dims.begin(); }
    auto end() { return dims.end(); }
    auto begin() const { return dims.begin(); }
    auto end() const { return dims.end(); }

    void push_back(size_t val) {
        dims.push_back(val);
    }

    void pop_back() { dims.pop_back(); }

    auto insert(typename std::vector<size_t>::iterator pos, size_t val) {
        return dims.insert(pos, val);
    }

    auto erase(typename std::vector<size_t>::iterator pos) {
        return dims.erase(pos);
    }

    auto erase(typename std::vector<size_t>::iterator first, typename std::vector<size_t>::iterator last) {
        return dims.erase(first, last);
    }

    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    auto data() const {
        return dims.data();
    }

    size_t& operator[](size_t i) { return dims[i]; }
    const size_t& operator[](size_t i) const { return dims[i]; }

    size_t size() const { return dims.size(); }

    size_t numel() const {
        size_t res = 1;
        for (auto x : dims) {
            res *= x;
        }
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
};

}

#endif // SHAPE_H