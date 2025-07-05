#include <iostream>
#include "linalg.hpp"
//#include "autodiff.hpp"
//#include "module.hpp"
//#include "solver.hpp"
//#include "dataloader.hpp"
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

    Tensor<> multi = Tensor<>::normal({10, 1000, 1000}, backend::BackendType::CpuMultiThread);
    Tensor<> cuda = multi.to(backend::BackendType::Cuda);

    {
        timing::ScopedTimer timer("multi");
        multi.assign(matmul(multi, multi));
    }
    {
        timing::ScopedTimer timer("cuda");
        cuda.assign(matmul(cuda, cuda));
    }
    cuda.assign(multi.to(backend::BackendType::Cuda) - cuda);
    cuda.max({1, 2}).print();
    cuda.min({1, 2}).print();
    // cuda.assign(cuda * cuda);
    // cuda.sum().print();
    // cuda.max().print();
    //(multi.to(backend::BackendType::Cuda) - cuda).sum().print();

    // Tensor<> cuda({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}, backend::BackendType::Cuda);
    // matmul(cuda, cuda).print();
}