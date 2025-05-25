#ifndef MODULE_H
#define MODULE_H

#include "autodiff.hpp"

using linalg::Tensor;
using autodiff::Expression;
using autodiff::ComputationGraph;

template <typename T>
class Module {
public:
    Module(std::shared_ptr<ComputationGraph> graph) : graph(graph) {
        if (graph == nullptr) {
            graph = std::make_shared<ComputationGraph>();
        }
    }

    virtual void train();
    virtual void eval();

    virtual Expression<T> forward(const Expression<T>& input);

    void backward() {
        graph->backward();
    }

    void zero_grad() {
        for (auto &w : weights) {
            w.grad[linalg::Range()] = 0;
        }
    }
protected:
    std::vector<Expression<T>> weights;
    std::vector<std::shared_ptr<Module<T>>> submodules;
    std::shared_ptr<ComputationGraph> graph;
};

template <typename T>
class Linear : public Module<T> {
public:
    Linear(int inputDim, int outputDim, std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(graph) {
        weights.push_back(Expression(Tensor<T>::normal({inputDim, outputDim}), graph));
        weights.push_back(Expression(Tensor<T>::normal({outputDim}), graph));
    }

    Expression<T> forward(const Expression<T>& input) override {
        return Expression<T>::matmul(input, weights[0]) + weights[1];
    }
protected:
    using Module<T>::weights;
    using Module<T>::submodules;
};
#endif // MODULE_H