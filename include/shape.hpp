#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include <memory>
#include <sstream>

// TODO: reimplement with SBO

struct Shape {
    std::vector<size_t> data;

    Shape() = default;
    Shape(std::vector<size_t> d) : data(std::move(d)) {}
    Shape(std::initializer_list<size_t> init) : data(init) {}

    operator std::vector<size_t>&() { return data; }
    operator const std::vector<size_t>&() const { return data; }

    // Forward vector methods
    auto front() const { return data.front(); }
    auto back() const { return data.back(); }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    void push_back(size_t val) {
        data.push_back(val);
    }

    void pop_back() { data.pop_back(); }

    auto insert(typename std::vector<size_t>::iterator pos, size_t val) {
        return data.insert(pos, val);
    }

    auto erase(typename std::vector<size_t>::iterator pos) {
        return data.erase(pos);
    }

    auto erase(typename std::vector<size_t>::iterator first, typename std::vector<size_t>::iterator last) {
        return data.erase(first, last);
    }

    bool operator==(const Shape& other) const {
        return data == other.data;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    size_t& operator[](size_t i) { return data[i]; }
    const size_t& operator[](size_t i) const { return data[i]; }

    size_t size() const { return data.size(); }

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

#endif // SHAPE_H