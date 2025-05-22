#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"

using linalg::Tensor;

namespace autograd {

template <typename T>
struct Node {
    Node() = delete;
    Node(const Tensor<T>& value, std::initializer_list<std::shared_ptr<Node<T>>> children, bool requiresGradient=false) : value(value), children(children) {
        //bool requiresGradient = false;
        // gradient is generated if requiresGradient is true or a child requires gradient
        for (auto p : children) {
            if (p->grad) {
                requiresGradient = true;
            }
            p->parentCount++;
        }

        if (requiresGradient) {
            grad = std::make_unique<Tensor<T>>(Tensor<T>::zeros(value.shape));
        }
    };

    virtual ~Node() = default;

    virtual void backward() = 0;

    void backward_recurse() {
        if (grad == nullptr)
            return;

        parentCount--;
        if (parentCount > 0)
            return;
        if (parentCount < 0) {
            throw std::invalid_argument("Unexpected parent count.");
        }

        backward();
        for (auto &p : children) {
            p->backward_recurse();
        }
    }

    Tensor<T> value;
    std::unique_ptr<Tensor<T>> grad;
    std::vector<std::shared_ptr<Node<T>>> children;
    // counts parents that we have not received gradients from
    int parentCount = 0;
};

template <typename T>
struct AddNode : public Node<T> {
    using Node<T>::value;
    using Node<T>::grad;
    using Node<T>::children;

    // static Tensor<T> forward(const Tensor<T>& a, const Tensor<T>& b) {
    //     return a + b;
    // }

    AddNode(std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b) : Node<T>(a->value + b->value, {a, b}) {}

    void backward() override {
        if (grad == nullptr)
            return;

        // handle broadcasted addition with sum
        for (int i = 0; i < 2; i++) {
            if (children[i]->grad) {
                *(children[i]->grad) += *grad;
            }
        }
    }
};

template <typename T>
struct SumNode : public Node<T> {
    using Node<T>::value;
    using Node<T>::grad;

    std::shared_ptr<Node<T>> a;

    SumNode(std::shared_ptr<Node<T>> a) : Node<T>(a->value.sum(), {a}), a(a) {}

    void backward() override {
        if (grad == nullptr)
            return;

        if (a->grad) {
            *(a->grad) += *grad;
        }
    }
};

template <typename T>
struct ValueNode : public Node<T> {
    using Node<T>::value;
    using Node<T>::grad;

    ValueNode(const Tensor<T>& t, bool requiresGradient) : Node<T>(t, {}, requiresGradient) {}

    void backward() override {}
};

// figure out where to put shared ptrs??

template <typename T = float>
class Expression {
public:
    Expression(const Tensor<T>& t, bool requiresGradient) {
        node = std::make_shared<ValueNode<T>>(t, requiresGradient);
    }

    Expression(std::shared_ptr<Node<T>> node) : node(node) {}

    Tensor<T>& value() {
        return node->value;
    }

    Tensor<T>& grad() {
        return *node->grad;
    }

    Expression operator+(const Expression& other) const {
        return Expression(std::make_shared<AddNode<T>>(node, other.node));
    }

    // only supports aggregate to scalar for now
    Expression sum() const {
        return Expression(std::make_shared<SumNode<T>>(node));
    }
    
    static Expression matmul(const Expression& a, const Expression& b) {
        // TODO: fill in
    }

    // goes in loss function, could make inheritance?
    void backward() {
        if (node->grad == nullptr)
            return;
        
        // assuming grad initialized to zero;
        (*node->grad) += 1;
        node->parentCount++;
        node->backward_recurse();
    }

    // reset gradients
    void zero_grad() {

    }
protected:
    std::shared_ptr<Node<T>> node;
};

} // namespace autograd

#endif // AUTOGRAD_H