#include <iostream>
#include "linalg.hpp"
#include "autodiff.hpp"
#include "module.hpp"
#include "solver.hpp"
#include "dataloader.hpp"
#include "loss.hpp"
#include "backend.hpp"
#include "timing.hpp"

using linalg::Tensor;
using linalg::Range;

void time_apply_binary() {
    backend::BackendGuard guard(backend::BackendType::CpuMultiThread);
    Tensor<> t = Tensor<>::normal({100000000});
    //Tensor<> single = t.to(backend::BackendType::CpuSingleThread);
    //single.assign(single * single);
    timing::Profiler::reset();
    for (size_t i = 0; i < 10; i++) {
        timing::ScopedProfiler timer("time");
        t * t;
        //Tensor<> res = t * t;
        //(res.to(backend::BackendType::CpuSingleThread) == single).astype<size_t>().sum().print();
    }
    timing::Profiler::report(true);
}

void time_reduction() {
    std::vector<Tensor<>> multi;
    for (size_t x : {100, 10000, 1000000, 100000000}) {
        backend::BackendGuard guard(backend::BackendType::CpuMultiThread);
        multi.push_back(Tensor<>::normal({x}));
    }

    std::vector<Tensor<>> single;
    for (auto &t : multi) {
        single.push_back(t.to(backend::BackendType::CpuSingleThread));
    }

    for (auto& t : single) {
        //backend::BackendGuard guard(backend::BackendType::CpuSingleThread);
        timing::ScopedTimer timer("single thread " + std::to_string(t.shape()[0]));

        t.sum();
        //t + t;
    }

    for (auto& t : multi) {
        //backend::BackendGuard guard(backend::BackendType::CpuMultiThread);
        timing::ScopedTimer timer("multi thread " + std::to_string(t.shape()[0]));

        t.sum();
        //t + t;
    }
}

void time_matmul() {
    Tensor<> multi = Tensor<float>::normal({1024, 1024}, backend::BackendType::CpuMultiThread);
    Tensor<> single = multi.to(backend::BackendType::CpuSingleThread);

    {
        timing::ScopedTimer timer("single thread");
        matmul(single, single);//.sum().print();
        //single.argmax(1).sum().print();
    }

    {
        timing::ScopedTimer timer("multi thread");
        matmul(multi, multi);//.sum().print();
        //multi.argmax(1).sum().print();
    }
}

int main() {
    time_apply_binary();
    std::cout << "Reduction" << std::endl;
    time_reduction();

    // std::cout << "\nMatmul" << std::endl;
    // time_matmul();

    return 0;
}