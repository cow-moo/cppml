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
    LinearReLU(size_t inputDim, size_t outputDim, std::string name="linear_relu", std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(name, graph) {
        W = this->register_weight("W", Tensor<T>::normal({inputDim, outputDim}, 0, sqrt(2.0f / inputDim), std::nullopt));
        b = this->register_weight("b", Tensor<T>::zeros({outputDim}));
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