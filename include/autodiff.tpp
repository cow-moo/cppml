#include "autodiff.hpp"

namespace autodiff {

void ComputationGraph::backward() {
    for (int i = tape.size() - 1; i >= 0; i--) {
        tape[i]();
    }
    tape.clear();
}

template <typename T>
Expression<T>::Expression(const Tensor<T>& value, std::string name, ComputationGraph* graph) : value_(value), name_(name), graph_(graph) {
    if (graph_) {
        grad_.emplace(Tensor<T>::zeros(this->value().shape(), value.backend_type()));
    }
}

template <typename T>
Expression<T>::Expression(const Tensor<T>& value, ComputationGraph* graph) : Expression(value, "unnamed", graph) {}

template <typename T>
Expression<T>::Expression(const Expression& other)
    : value_(other.value_), grad_(other.grad_), name_(other.name_), graph_(other.graph_) {}

template <typename T>
Expression<T>& Expression<T>::operator=(const Expression& other) {
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

template <typename T>
Expression<T>& Expression<T>::rename_(const std::string& name) {
    name_ = name;
    return *this;
}

template <typename T>
void Expression<T>::backward() {
    assert(graph_ != nullptr);
    assert(grad_->shape().size() == 0);

    (*grad_) += 1;
    graph_->backward();
}

template <typename T>
template <typename... Args>
Expression<T> Expression<T>::operator[](Args&&... args) const {
    Expression res(*this);
    res.value_->assign(value().at(std::forward<Args>(args)...));
    if (requires_grad())
        res.grad_->assign(grad().at(std::forward<Args>(args)...));
    return res;
}

template <typename T>
Expression<T> Expression<T>::operator+(const Expression& other) const {
    Expression res(value() + other.value(), "+", graph_ ? other.graph_ : other.graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([grad = *res.grad_, aGrad = grad_, bGrad = other.grad_]() mutable {
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

template <typename T>
Expression<T> Expression<T>::operator-(const Expression& other) const {
    Expression res(value() - other.value(), "-", graph_ ? graph_ : other.graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([grad = *res.grad_, aGrad = grad_, bGrad = other.grad_]() mutable {
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

template <typename T>
Expression<T> Expression<T>::operator*(const Expression& other) const {
    Expression res(value() * other.value(), "*", graph_ ? graph_ : other.graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([grad = *res.grad_, aGrad = grad_, bGrad = other.grad_, aValue = value(), bValue = other.value()]() mutable {
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

template <typename T>
Expression<T> Expression<T>::operator/(const Expression& other) const {
    Expression res(value() / other.value(), "/", graph_ ? graph_ : other.graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([grad = *res.grad_, aGrad = grad_, bGrad = other.grad_, aValue = value(), bValue = other.value()]() mutable {
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

template <typename T>
Expression<T> Expression<T>::operator*(T other) {
    Expression res(value() * other, "*", graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_, other]() mutable {
            grad += resGrad * other;
        });
    }

    return res;
}

template <typename T>
Expression<T> Expression<T>::operator/(T other) {
    return ((*this) * (1 / other)).rename_("/");
}

template <typename T>
Expression<T> Expression<T>::matmul(const Expression& other) const {
    Expression res(linalg::matmul(value(), other.value()), "matmul", graph_ ? graph_ : other.graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([grad = *res.grad_, aGrad = grad_, bGrad = other.grad_, aValue = value(), bValue = other.value()]() mutable {
            if (aGrad) {
                *aGrad += linalg::matmul(grad, bValue.T()).broadcast_reduce_to(aGrad->shape());
            }
            if (bGrad) {
                *bGrad += linalg::matmul(aValue.T(), grad).broadcast_reduce_to(bGrad->shape());
            }
        });
    }

    return res;
}

template <typename T>
Expression<T> Expression<T>::sum() const {
    Expression res(value().sum(), "sum", graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_]() mutable {
            grad += resGrad;
        });
    }

    return res;
}

template <typename T>
Expression<T> Expression<T>::relu() const {
    Expression res(value().relu(), "relu", graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([resGrad = *res.grad_, grad = *grad_, value = this->value()]() mutable {
            grad += (value > 0.0f).template astype<T>() * resGrad;
        });
    }

    return res;
}

template <typename T>
Expression<T> Expression<T>::softmax() const {
    Expression res(value().softmax(), "softmax", graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([resGrad = *res.grad_, resValue = res.value(), grad = *grad_]() mutable {
            Tensor<T> scratch = (resGrad * resValue).sum({-1}).unsqueeze(resGrad.shape().size() - 1);
            grad += resValue * (resGrad - scratch);
        });
    }

    return res;
}

template <typename T>
Expression<T> Expression<T>::log_softmax() const {
    Expression res(value().log_softmax(), "log_softmax", graph_);

    if (res.graph_) {
        res.graph_->tape.push_back([resGrad = *res.grad_, resValue = res.value(), grad = *grad_]() mutable {
            Tensor<T> sumGrad = resGrad.sum({-1}).unsqueeze(resGrad.shape().size() - 1);
            grad += resGrad - resValue.exp() * sumGrad;
        });
    }

    return res;
}

template <typename T>
void Expression<T>::print() const {
    std::cout << name_ << std::endl;
    std::cout << "value ";
    value().print();
    if (grad_) {
        std::cout << "grad  ";
        grad_->print();
    }
}

}