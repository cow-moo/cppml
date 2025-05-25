#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <vector>
#include <functional>
#include "tensor.hpp"

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

    Expression(const Tensor<T>& value, ComputationGraph* graph=nullptr) : value(value), graph(graph) {
        if (graph) {
            grad = Tensor<T>::zeros(value.shape);
        }
    }

    // Should only be called on the final loss function
    void backward() {
        assert(graph != nullptr);
        assert(grad->shape.size() == 0);

        (*grad) += 1;
        graph->backward();
    }

    // undefined behavior if a and b come from different computation graphs
    friend Expression operator+(const Expression& a, const Expression& b) {
        Expression res(a.value + b.value, a.graph == nullptr ? b.graph : a.graph);

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad]() mutable {
                if (aGrad) {
                    *aGrad += grad.broadcast_reduce_to(aGrad->shape);
                }
                if (bGrad) {
                    *bGrad += grad.broadcast_reduce_to(bGrad->shape);
                }
            });
        }

        return res;
    }

    // undefined behavior if a and b come from different computation graphs
    friend Expression operator*(const Expression& a, const Expression& b) {
        Expression res(a.value * b.value, a.graph == nullptr ? b.graph : a.graph);

        if (res.graph) {
            res.graph->tape.push_back([grad = *res.grad, aGrad = a.grad, bGrad = b.grad, aValue = a.value, bValue = b.value]() mutable {
                if (aGrad) {
                    *aGrad += (grad * bValue).broadcast_reduce_to(aGrad->shape);
                }
                if (bGrad) {
                    *bGrad += (grad * aValue).broadcast_reduce_to(bGrad->shape);
                }
            });
        }

        return res;
    }

    Expression sum() const {
        Expression res(value.sum(), graph);

        if (res.graph) {
            res.graph->tape.push_back([resGrad = *res.grad, grad = *grad]() mutable {
                grad += resGrad;
            });
        }

        return res;
    }

    void print() const {
        std::cout << "value ";
        value.print();
        if (grad) {
            std::cout << "grad  ";
            grad->print();
        }
    }
};

} // namespace autodiff

#endif // AUTODIFF_H