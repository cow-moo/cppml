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
    Tensor<> multi = Tensor<>::normal({10, 1000000}, backend::BackendType::CpuMultiThread);
    Tensor<> single = multi.to(backend::BackendType::CpuSingleThread);
    Tensor<> cuda = multi.to(backend::BackendType::Cuda);

    {
        timing::ScopedTimer timer("multi");
        // multi.sum({1}).print();
        multi.argmax().print();
    }
    {
        timing::ScopedTimer timer("single");
        // single.sum({1}).print();
        single.argmax().print();
    }
    {
        timing::ScopedTimer timer("cuda");
        // cuda.sum({1}).print();
        cuda.argmax().print();
    }

    // Tensor<> t({1, 2, 3}, backend::BackendType::Cuda);
    // t.sum().print();
}