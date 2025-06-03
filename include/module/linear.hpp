#ifndef MODULE_LINEAR_H
#define MODULE_LINEAR_H

#include "base.hpp"
#include "autodiff.hpp"

using linalg::Tensor;
using autodiff::Expression;
using autodiff::ComputationGraph;

namespace module {

template <typename T = float>
class LinearReLU : public Module<T> {
public:
    LinearReLU(size_t inputDim, size_t outputDim, std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(graph) {
        W = this->register_weight(Tensor<T>::normal({inputDim, outputDim}, 0, sqrt(2 / inputDim)), "W");
        b = this->register_weight(Tensor<T>::zeros({outputDim}), "b");
    }

    Expression<T> forward(const Expression<T>& input) override {
        return relu(matmul(input, W) + b);
    }

private:
    Expression<T> W;
    Expression<T> b;
};

}

#endif // MODULE_LINEAR_H