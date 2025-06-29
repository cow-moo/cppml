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

    multi.sum({1}).print();
    single.sum({1}).print();
}