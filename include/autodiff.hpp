#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <vector>
#include <functional>
#include "linalg.hpp"

namespace autodiff {

using linalg::Tensor;
using linalg::Shape;

struct ComputationGraph {
    // Tape is in topological sort order of computation graph DAG assuming graph is
    // constructed in order on a single thread
    std::vector<std::function<void()>> tape;

    void backward();
};

template <typename T = float>
class Expression {
public:
    Expression() = default;

    Expression(const Tensor<T>& value, std::string name = "unnamed", ComputationGraph* graph = nullptr);

    Expression(const Tensor<T>& value, ComputationGraph* graph);

    Expression(const Expression& other);

    // Rebinding assignment
    Expression& operator=(const Expression& other);

    Tensor<T>& value() { return *value_; }
    const Tensor<T>& value() const { return *value_; }

    Tensor<T>& grad() { return *grad_; }
    const Tensor<T>& grad() const { return *grad_; }

    bool requires_grad() const { return grad_.has_value(); }

    const Shape& shape() const { return value().shape(); }

    Expression& rename_(const std::string& name);

    // Should only be called on the final loss function
    void backward();

    template <typename... Args>
    Expression operator[](Args&&... args) const;

    // undefined behavior if a and b come from different computation graphs
    Expression operator+(const Expression& other) const;
    Expression operator-(const Expression& other) const;
    Expression operator*(const Expression& other) const;
    Expression operator/(const Expression& other) const;

    Expression operator-() { return (*this) * -1; }

    Expression operator*(T other);
    Expression operator/(T other);

    Expression matmul(const Expression& other) const;

    Expression relu() const;

    Expression sum() const;

    // Softmax on last dimension
    Expression softmax() const;
    Expression log_softmax() const;

    Expression gather(const Tensor<size_t>& idxs) const;

    void print() const;

private:
    // Ops (aside from assignment) should never reassign these
    std::optional<Tensor<T>> value_;
    std::optional<Tensor<T>> grad_;
    std::string name_;
    ComputationGraph* graph_;
};

template <typename T>
Expression<T> matmul(const Expression<T>& a, const Expression<T>& b) { return a.matmul(b); }

template <typename T>
Expression<T> relu(const Expression<T>& expr) { return expr.relu(); }

template <typename T>
Expression<T> sum(const Expression<T>& expr) { return expr.sum(); }

} // namespace autodiff

#include "autodiff.tpp"

#endif // AUTODIFF_H