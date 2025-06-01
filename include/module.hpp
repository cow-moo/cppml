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
    Module(std::shared_ptr<ComputationGraph> graph) : graph_(graph) {
        if (graph_ == nullptr) {
            graph_ = std::make_shared<ComputationGraph>();
        }
    }

    virtual void train() {}
    virtual void eval() {}

    virtual Expression<T> forward(const Expression<T>& input) = 0;

    void backward() {
        graph_->backward();
    }

    void zero_grad() {
        for (auto &w : weights_) {
            w.grad[linalg::Range()] = 0;
        }
    }

    std::vector<Expression<T>> weights() {
        std::vector<Expression<T>> res(weights_);
        for (auto &m : submodules_) {
            auto temp = m->weights();
            res.insert(res.end(), temp.begin(), temp.end());
        }
        return res;
    }
protected:
    std::vector<Expression<T>> weights_;
    std::vector<std::shared_ptr<Module<T>>> submodules_;
    std::shared_ptr<ComputationGraph> graph_;
};

template <typename T = float>
class Linear : public Module<T> {
public:
    Linear(size_t inputDim, size_t outputDim, std::shared_ptr<ComputationGraph> graph=nullptr) : Module<T>(graph) {
        weights_.push_back(Expression(Tensor<T>::normal({inputDim, outputDim}), graph_.get()));
        weights_.push_back(Expression(Tensor<T>::normal({outputDim}), graph_.get()));
    }

    Expression<T> forward(const Expression<T>& input) override {
        return matmul(input, weights_[0]) + weights_[1];
    }
protected:
    using Module<T>::weights_;
    using Module<T>::submodules_;
    using Module<T>::graph_;
};

} // namespace module

#endif // MODULE_H