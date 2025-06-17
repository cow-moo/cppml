#include <iostream>
#include "dataloader.hpp"
#include "tensor.hpp"
#include "autodiff.hpp"
#include "module.hpp"
#include "solver.hpp"
#include "loss.hpp"

using linalg::Tensor;
using linalg::Shape;
using autodiff::Expression;

int main() {
    Tensor<> x = {0, 1, 2, 3, 4};
    x.assign(x.reshape({5, 1}));

    Tensor<> y = {3, 5, 7, 9, 11};
    y.assign(y.reshape({5, 1}));

    module::Linear<> linear(1, 1);
    solver::GradientDescent gd(linear.weights(), 0.1);

    for (int epoch = 0; epoch < 100; epoch++) {
        auto yPred = linear(x);
        auto loss = loss::mse(yPred, y);
        loss.backward();
        gd.step();
        gd.zero_grad();
        loss.value().print();
        yPred.value().print();
    }
    
    for (auto &w : linear.weights()) {
        w.print();
    }
}