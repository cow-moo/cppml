#ifndef MODULE_MLP_H
#define MODULE_MLP_H

#include "base.hpp"
#include "autodiff.hpp"

using linalg::Tensor;
using autodiff::Expression;
using autodiff::ComputationGraph;

namespace module {

template <typename T = float>
class MLP : public module::Module<T> {
public:
    MLP(const std::vector<size_t>& dims, std::string name="mlp", std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(name, graph) {
        assert(dims.size() >= 2);

        for (size_t i = 1; i < dims.size() - 1; i++) {
            hiddenLayers.push_back(
                this->template register_module<module::LinearReLU<T>>(
                    "l" + std::to_string(i - 1), dims[i - 1], dims[i]
                )
            );
        }
        outputLayer = this->template register_module<module::Linear<T>>(
            "l" + std::to_string(dims.size() - 1), dims[dims.size() - 2], dims[dims.size() - 1]
        );
    }

    Expression<T> forward(const Expression<T>& input) override {
        Expression<T> res = input;
        for (auto &layer : hiddenLayers) {
            res = layer->forward(res);
        }
        return outputLayer->forward(res);
    }

private:
    std::vector<std::shared_ptr<module::LinearReLU<T>>> hiddenLayers;
    std::shared_ptr<module::Linear<T>> outputLayer;
};

}

#endif // MODULE_MLP_H