#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <vector>
#include <functional>
#include "tensor.hpp"
#include "shape.hpp"

namespace autodiff {

using linalg::Tensor;
using linalg::Shape;

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
    Expression() = default;

    Expression(const Tensor<T>& value, std::string name="unnamed", ComputationGraph* graph=nullptr) : value_(value), name_(name), graph_(graph) {
        if (graph_) {
            grad_.emplace(Tensor<T>::zeros(this->value().shape()));
        }
    }

    Expression(const Tensor<T>& value, ComputationGraph* graph) : Expression(value, "unnamed", graph) {}

    Expression(const Expression& other) 
        : value_(other.value_), grad_(other.grad_), name_(other.name_), graph_(other.graph_) {}

    // Rebinding assignment
    Expression& operator=(const Expression& other) {
        if (value_)
            value_->assign(other.value());
        else
            value_ = other.value();
        grad_.reset();
        if (other.grad_) {
            grad_.emplace(*other.grad_);
        }
        graph_ = other.graph_;
        name_ = other.name_;
        return *this;
    }

    Tensor<T>& value() {
        return *value_;
    }

    const Tensor<T>& value() const {
        return *value_;
    }

    Tensor<T>& grad() {
        return *grad_;
    }

    const Tensor<T>& grad() const {
        return *grad_;
    }

    bool requires_grad() const {
        return grad_.has_value();
    }

    const Shape& shape() const {
        return value().shape();
    }

    // Should only be called on the final loss function
    void backward() {
        assert(graph_ != nullptr);
        assert(grad_->shape().size() == 0);

        (*grad_) += 1;
        graph_->backward();
    }

    template <typename... Args>
    Expression operator[](Args&&... args) const {
        Expression res(*this);
        res.value_->assign(value().at(std::forward<Args>(args)...));
        if (requires_grad())
           res.grad_->assign(grad().at(std::forward<Args>(args)...));
        return res;
    }

    // undefined behavior if a and b come from different computation graphs
    friend Expression operator+(const Expression& a, const Expression& b) {
        Expression res(a.value() + b.value(), "+", a.graph_ == nullptr ? b.graph_ : a.graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([grad = *res.grad_, aGrad = a.grad_, bGrad = b.grad_]() mutable {
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
        Expression res(a.value() - b.value(), "-", a.graph_ == nullptr ? b.graph_ : a.graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([grad = *res.grad_, aGrad = a.grad_, bGrad = b.grad_]() mutable {
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
        Expression res(a.value() * b.value(), "*", a.graph_ == nullptr ? b.graph_ : a.graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([grad = *res.grad_, aGrad = a.grad_, bGrad = b.grad_, aValue = a.value(), bValue = b.value()]() mutable {
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
        Expression res(a.value() / b.value(), "/", a.graph_ == nullptr ? b.graph_ : a.graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([grad = *res.grad_, aGrad = a.grad_, bGrad = b.grad_, aValue = a.value(), bValue = b.value()]() mutable {
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

    Expression operator-() {
        return (*this) * -1;
    }

    Expression operator*(T other) {
        Expression res(value() * other, "*", graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_, other]() mutable {
                grad += resGrad * other;
            });
        }

        return res;
    }

    Expression operator/(T other) {
        return (*this) * (1 / other);
    }

    friend Expression matmul(const Expression& a, const Expression& b) {
        Expression res(matmul(a.value(), b.value()), "matmul", a.graph_ == nullptr ? b.graph_ : a.graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([grad = *res.grad_, aGrad = a.grad_, bGrad = b.grad_, aValue = a.value(), bValue = b.value()]() mutable {
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
        Expression res(value().sum(), "sum", graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_]() mutable {
                grad += resGrad;
            });
        }

        return res;
    }

    Expression relu() const {
        Expression res(value().relu(), "relu", graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_, value = this->value()]() mutable {
                grad += (value > 0.0f).template astype<T>() * resGrad;
            });
        }

        return res;
    }

    // Softmax on last dimension
    Expression softmax() const {
        Expression res(value().softmax(), "softmax", graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([resGrad = *res.grad_, resValue = res.value(), grad = *grad_]() mutable {
                Tensor<T> scratch = (resGrad * resValue).sum({-1}).unsqueeze(resGrad.shape().size() - 1);
                grad += resValue * (resGrad - scratch);
            });
        }

        return res;
    }

    Expression log_softmax() const {
        Expression res(value().log_softmax(), "log_softmax", graph_);

        if (res.graph_) {
            res.graph_->tape.push_back([resGrad = *res.grad_, resValue = res.value(), grad = *grad_]() mutable {
                Tensor<T> sumGrad = resGrad.sum({-1}).unsqueeze(resGrad.shape().size() - 1);
                grad += resGrad - resValue.exp() * sumGrad;
            });
        }

        return res;
    }

    void print() const {
        std::cout << name_ << std::endl;
        std::cout << "value ";
        value().print();
        if (grad_) {
            std::cout << "grad  ";
            grad_->print();
        }
    }

private:
    std::optional<Tensor<T>> value_;
    std::optional<Tensor<T>> grad_;
    std::string name_;
    ComputationGraph* graph_;
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