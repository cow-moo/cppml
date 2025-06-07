#include <iostream>
#include "tensor.hpp"
#include "autodiff.hpp"
#include "module.hpp"
#include "solver.hpp"
#include "dataloader.hpp"
#include "loss.hpp"
#include "backend.hpp"

using linalg::Tensor;
using linalg::Range;
using linalg::Shape;
using autodiff::Expression;
using autodiff::ComputationGraph;

int main() {
    Tensor<float> a({{1, 20, 3}, {4, 5, 6}});
    //Tensor<float> b({{1, 3, 5}, {7, 9, 11}});
    Tensor<float> b({5, 1, 50});
    Tensor<float> c({10, 100});
    c.assign(c.reshape(Shape{2, 1}));

    // (a + c).print();
    // (a + b).print();
    // (b + c).print();

    //a.max({1}, false).print();
    // a.softmax(0).print();
    // a.log_softmax(0).exp().print();

    a.argmin(1, false).print();

    a.min({1}).print();

    //(b * a).print();
    //a *= b;
    //a.print();
    // b.print();
    // a /= 4.0f;
    // a.print();
    // (a == b).print();
    // a.print();
    // b.print();
    // Tensor<float> c = a.exp();
    // c.print();
    // c.log().print();

    //std::cout << sizeof(backend::CpuSingleThreadBuffer<float>) << " " << sizeof(backend::DeviceBuffer<float>) << std::endl;

    // Shape shape({2, 3});
    // Tensor<float> a(shape);
    // Tensor<float> b(shape);
    // Tensor<float> c({1, 1});
    // Tensor<float> d(shape);

    // Tensor<float> e = c + c;
    // e.print();

    return 0;
}