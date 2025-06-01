#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <vector>
#include <functional>
#include "tensor.hpp"
#include "shape.hpp"

using linalg::Tensor;

namespace autodiff {

class ComputationGraph {
public:
    std::vector<std::function<void()>> tape;

    void backward() {
        for (int i = tape.size() - 1; i >= 0; i--) {
            tape[i]();
        }
        tape.clear();
    }
};

template <typename T = float>
class Expression {
public:
    Tensor<T> value;
    std::optional<Tensor<T>> grad;
    ComputationGraph* graph; // could use weak_ptr but it doesn't have easy nullptr
    std::string name;

    Expression(const Tensor<T>& value, ComputationGraph* graph=nullptr, std::string name="unnamed") : value(value), graph(graph), name(name) {
        if (graph) {
            grad.emplace(Tensor<T>::zeros(value.shape()));
        }
    }

    Expression(const Expression& other) 
        : value(other.value), grad(other.grad), graph(other.graph), name(other.name) {}

    // Rebinding assignment
    Expression& operator=(const Expression& other) {
        value.assign(other.value);
        grad.reset();
        if (other.grad) {
            grad.emplace(*other.grad);
        }
        name = other.name;
        return *this;
    }

    const Shape& shape() const {
        return value.shape();
    }

    // Should only be called on the final loss function
    void backward() {
        assert(graph != nullptr);
        assert(grad->shape().size() == 0);

        (*grad) += 1;
        graph->backward();
    }

    // undefined behavior if a and b come from different computation graphs
    friend Expression operator+(const Expression& a, const Expression& b) {
        Expression res(a.value + b.value, a.graph == nullptr ? b.graph : a.graph, "+");

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad]() mutable {
                if (aGrad) {
                    *aGrad += grad.broadcast_reduce_to(aGrad->shape());
                }
                if (bGrad) {
                    *bGrad += grad.broadcast_reduce_to(bGrad->shape());
                }
            });
        }

        return res;
    }

    friend Expression operator-(const Expression& a, const Expression& b) {
        Expression res(a.value - b.value, a.graph == nullptr ? b.graph : a.graph, "-");

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad]() mutable {
                if (aGrad) {
                    *aGrad += grad.broadcast_reduce_to(aGrad->shape());
                }
                if (bGrad) {
                    *bGrad -= grad.broadcast_reduce_to(bGrad->shape());
                }
            });
        }

        return res;
    }

    // undefined behavior if a and b come from different computation graphs
    friend Expression operator*(const Expression& a, const Expression& b) {
        Expression res(a.value * b.value, a.graph == nullptr ? b.graph : a.graph, "*");

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad, aValue = a.value, bValue = b.value]() mutable {
                if (aGrad) {
                    *aGrad += (grad * bValue).broadcast_reduce_to(aGrad->shape());
                }
                if (bGrad) {
                    *bGrad += (grad * aValue).broadcast_reduce_to(bGrad->shape());
                }
            });
        }

        return res;
    }

    friend Expression operator/(const Expression& a, const Expression& b) {
        Expression res(a.value / b.value, a.graph == nullptr ? b.graph : a.graph, "/");

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad, aValue = a.value, bValue = b.value]() mutable {
                if (aGrad) {
                    *aGrad += (grad / bValue).broadcast_reduce_to(aGrad->shape());
                }
                if (bGrad) {
                    *bGrad += (-grad * aValue / bValue / bValue).broadcast_reduce_to(bGrad->shape());
                }
            });
        }

        return res;
    }

    Expression operator*(T other) {
        Expression res(value * other, graph, "*");

        if (res.graph) {
            res.graph->tape.push_back([resGrad = *res.grad, grad = *grad, other]() mutable {
                grad += resGrad * other;
            });
        }

        return res;
    }

    Expression operator/(T other) {
        return (*this) * (1 / other);
    }

    friend Expression matmul(const Expression& a, const Expression& b) {
        Expression res(matmul(a.value, b.value), a.graph == nullptr ? b.graph : a.graph, "matmul");

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad, aValue = a.value, bValue = b.value]() mutable {
                if (aGrad) {
                    *aGrad += matmul(grad, bValue.T()).broadcast_reduce_to(aGrad->shape());
                }
                if (bGrad) {
                    *bGrad += matmul(aValue.T(), grad).broadcast_reduce_to(bGrad->shape());
                }
            });
        }

        return res;
    }

    Expression sum() const {
        Expression res(value.sum(), graph, "sum");

        if (res.graph) {
            res.graph->tape.push_back([resGrad = *res.grad, grad = *grad]() mutable {
                grad += resGrad;
            });
        }

        return res;
    }

    Expression relu() const {
        Expression res(value.apply_unary([](T val) { return val > 0 ? val : 0; }), graph, "relu");

        if (res.graph) {
            res.graph->tape.push_back([resGrad = *res.grad, grad = *grad, value = this->value]() mutable {
                grad += value.apply_unary([](T val) { return val > 0 ? 1 : 0; }) * resGrad;
            });
        }

        return res;
    }

    void print() const {
        std::cout << name << std::endl;
        std::cout << "value ";
        value.print();
        if (grad) {
            std::cout << "grad  ";
            grad->print();
        }
    }
};

template <typename T>
Expression<T> sum(const Expression<T>& expr) {
    return expr.sum();
}

template <typename T>
Expression<T> relu(const Expression<T>& expr) {
    return expr.relu();
}

} // namespace autodiff

#endif // AUTODIFF_H