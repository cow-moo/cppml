#ifndef MODULE_H
#define MODULE_H

#include "autograd.hpp"

using linalg::Tensor;
using autograd::Expression;

template <typename T>
class Module {
public:
    virtual void train();
    virtual void eval();

    virtual Expression<T> forward(const Expression<T>& input);
protected:
    std::vector<Expression<T>> weights;
    std::vector<std::unique_ptr<Module<T>>> submodules;
};

template <typename T>
class Linear : public Module<T> {
public:
    Linear(int inputDim, int outputDim) {
        weights.push_back(Expression(Tensor<T>::normal({inputDim, outputDim}), true));
        weights.push_back(Expression(Tensor<T>::normal({outputDim}), true));
    }

    Expression<T> forward(const Expression<T>& input) override {
        return Expression<T>::matmul(input, weights[0]) + weights[1];
    }
protected:
    using Module<T>::weights;
    using Module<T>::submodules;
};
#endif // MODULE_H