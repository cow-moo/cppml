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

    virtual void step() = 0;

protected:
    std::vector<Expression<T>> weights_;
};

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    GradientDescent(const std::vector<Expression<T>>& weights, float learningRate) : Optimizer<T>(weights), learningRate(learningRate) {

    }

    void step() override {
        for (auto& expr : weights_) {
            if (expr.grad)
                expr.value += learningRate * (*expr.grad);
        }
    }

protected:
    using Optimizer<T>::weights_;

    float learningRate;
};


/* Loss Functions */

// class LossFunction {

// };

// class MSE : public LossFunction {

// };

} // namespace solver

#endif // SOLVER_H