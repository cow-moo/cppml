#include <iostream>
#include "linalg.hpp"
#include "autodiff.hpp"
//#include "module.hpp"
//#include "solver.hpp"
#include "data.hpp"
//#include "loss.hpp"
#include "backend.hpp"
#include "timing.hpp"
#include "backend/cuda_buffer.hpp"
// #include <cuda_runtime.h>

using linalg::Tensor;
using linalg::Range;
using linalg::Shape;
// using autodiff::Expression;
// using autodiff::ComputationGraph;

int main() {
    // Tensor<> multi = Tensor<>::normal({10, 1000000}, backend::BackendType::CpuMultiThread);
    // Tensor<> single = multi.to(backend::BackendType::CpuSingleThread);
    // Tensor<> cuda = multi.to(backend::BackendType::Cuda);

    // {
    //     timing::ScopedTimer timer("multi");
    //     // multi.sum({1}).print();
    //     multi.argmax().print();
    // }
    // {
    //     timing::ScopedTimer timer("single");
    //     // single.sum({1}).print();
    //     single.argmax().print();
    // }
    // {
    //     timing::ScopedTimer timer("cuda");
    //     // cuda.sum({1}).print();
    //     cuda.argmax().print();
    // }

    // Tensor<> multi = Tensor<>::normal({10, 1000, 1000}, backend::BackendType::CpuMultiThread);
    // Tensor<> cuda({0});
    // {
    //     timing::ScopedTimer timer("assign");
    //     cuda.assign(multi.to(backend::BackendType::Cuda));
    // }

    // {
    //     timing::ScopedTimer timer("multi");
    //     multi.assign(matmul(multi, multi));
    // }
    // {
    //     timing::ScopedTimer timer("cuda");
    //     cuda.assign(matmul(cuda, cuda));
    // }
    // cuda.assign(multi.to(backend::BackendType::Cuda) - cuda);
    // cuda.max({1, 2}).print();
    // cuda.min({1, 2}).print();

    // backend::BackendGuard guard(backend::BackendType::Cuda);
    // data::MNISTDataset train("../datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte", 
    //                          "../datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
    
    // std::cout << "Dataset ready" << std::endl;
    // data::DataLoader dl(std::move(train), 64);
    // dl.shuffle();
    // auto it = dl.begin();
    // std::get<1>(*it).print();

    backend::BackendGuard guard(backend::BackendType::Cuda);
    // Tensor<> a({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    // Tensor<size_t> b({{0, 0}, {1, 0}, {0, 2}});
    // b.print();
    // a.gather(b).print();

    // Tensor<> a({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    // Tensor<size_t> b({{0, 1, 3}, {1, 0, 2}, {0, 2, 2}});

    // a.scatter(b, 4).print();

    autodiff::ComputationGraph graph;
    autodiff::Expression<> a(linalg::Tensor<>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}), &graph);
    linalg::Tensor<size_t> b({0, 2, 0});
    b.assign(b.unsqueeze(1));

    auto x = a.gather(b).sum();
    x.backward();
    a.print();
}