#include <iostream>
#include "dataloader.hpp"
#include "linalg.hpp"
#include "autodiff.hpp"
#include "module.hpp"
#include "solver.hpp"
#include "loss.hpp"

using linalg::Tensor;
using linalg::Shape;
using autodiff::Expression;

int main() {
    Tensor<> x = Tensor<>::zeros({100, 1});
    Tensor<> y = Tensor<>::zeros({100, 1});
    for (int i = 0; i < 100; ++i) {
        float v = -1.0f + 2.0f * i / 99;
        x[i][0] = v;
        y[i][0] = v * v;
    }

    // x[Range(5)].print();
    // y[Range(5)].print();

    module::MLP<> model({1, 8, 1});
    solver::GradientDescent gd(model.weights(), 0.01);

    for (int epoch = 0; epoch < 1000; epoch++) {
        auto yPred = model(x);
        auto loss = loss::mse(yPred, y);
        loss.backward();
        gd.step();
        gd.zero_grad();
        if (epoch % 100 == 0)
            loss.value().print();
        //yPred.value.print();
    }
    loss::mse(model(x), y).print();

    //std::cout << "y pred" << std::endl;
    //(model(x).value() - y).print();
    //std::cout << "y" << std::endl;
    //y.print();
    
    for (auto &w : model.weights()) {
        w.print();
    }
}