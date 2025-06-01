#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

namespace solver {

template <typename T>
class Optimizer {
public:
    Optimizer(const std::vector<Expression<T>>& weights) {
        weights_ = weights;
    }

    void zero_grad() {
        for (auto &w : weights_) {
            (*w.grad) = 0;
        }
    }

    virtual void step() = 0;

protected:
    std::vector<Expression<T>> weights_;
};

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    GradientDescent(const std::vector<Expression<T>>& weights, float learningRate) 
        : Optimizer<T>(weights), learningRate_(learningRate) {}

    void step() override {
        for (auto& expr : weights_) {
            if (expr.grad)
                expr.value -= learningRate_ * (*expr.grad);
        }
    }

protected:
    using Optimizer<T>::weights_;

    float learningRate_;
};


/* Loss Functions */

// class LossFunction {

// };

// class MSE : public LossFunction {

// };

} // namespace solver

#endif // SOLVER_H