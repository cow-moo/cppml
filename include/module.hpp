#ifndef MODULE_H
#define MODULE_H

#include "autodiff.hpp"

using linalg::Tensor;
using autodiff::Expression;
using autodiff::ComputationGraph;

namespace module {

template <typename T>
class Module {
public:
    Module(std::shared_ptr<ComputationGraph> graph=nullptr) : graph_(graph) {
        if (graph_ == nullptr) {
            graph_ = std::make_shared<ComputationGraph>();
        }
    }

    virtual ~Module() = default;

    virtual void train() {}
    virtual void eval() {}

    virtual Expression<T> forward(const Expression<T>& input) = 0;

    Expression<T> operator()(const Expression<T>& input) {
        return forward(input);
    }

    std::vector<Expression<T>> weights() {
        std::vector<Expression<T>> res(weights_);
        for (auto &m : submodules_) {
            auto temp = m->weights();
            res.insert(res.end(), temp.begin(), temp.end());
        }
        return res;
    }

    Expression<T> register_weight(const Tensor<T>& val, std::string name="unnamed") {
        assert(graph_);
        weights_.push_back(Expression(val, graph_.get(), name));
        return weights_.back();
    }

    template <typename ModuleType, typename... Args>
    std::shared_ptr<ModuleType> register_module(Args&&... args) {
        assert(graph_);
        auto m = std::make_shared<ModuleType>(std::forward<Args>(args)..., graph_);
        submodules_.push_back(m);
        return m;
    }

private:
    std::vector<Expression<T>> weights_;
    std::vector<std::shared_ptr<Module<T>>> submodules_;
    std::shared_ptr<ComputationGraph> graph_;
};

template <typename T = float>
class Linear : public Module<T> {
public:
    Linear(size_t inputDim, size_t outputDim, std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(graph) {
        W = this->register_weight(Tensor<T>::normal({inputDim, outputDim}, 0, sqrt(1 / inputDim)), "W");
        b = this->register_weight(Tensor<T>::zeros({outputDim}), "b");
    }

    Expression<T> forward(const Expression<T>& input) override {
        return matmul(input, W) + b;
    }

private:
    Expression<T> W;
    Expression<T> b;
};

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

} // namespace module

#endif // MODULE_H