#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

namespace solver {

template <typename T>
class Optimizer {
public:
    Optimizer(Module module) {
        //TODO: do some kind of recursion to get all weights?
    }

    virtual void step();

protected:
    std::vector<Expression<T>> weights;
};

template <typename T>
class GradientDescent : public Optimizer<T> {
public:
    GradientDescent(float learningRate) : learningRate(learningRate) {

    }

    void step() override {
        for (auto& expr : weights) {
            expr.value() += learningRate * expr.grad();
        }
    }

protected:
    using Optimizer<T>::weights;

    float learningRate;
};


/* Loss Functions */

class LossFunction {

};

class MSE : public LossFunction {

};

} // namespace solver

#endif // SOLVER_H