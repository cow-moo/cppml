#ifndef LOSS_H
#define LOSS_H

#include "autodiff.hpp"

namespace loss {

template <typename T>
Expression<T> mse(const Expression<T>& pred, const Tensor<T>& target) {
    Expression<T> diff = pred - target;
    return (sum(diff * diff) / diff.value().numel()).rename_("mse");
}

template <typename T>
Expression<T> cross_entropy_logits(const Expression<T>& logits, const Tensor<T>& target) {
    return (sum(logits.log_softmax() * Expression<T>(target)) / (-(float)target.shape()[0])).rename_("cross_entropy_logits");
}

template <typename T>
Expression<T> cross_entropy_logits(const Expression<T>& logits, const Tensor<size_t>& target) {
    return (sum(logits.log_softmax() * Expression<T>(target)) / (-(float)target.shape()[0])).rename_("cross_entropy_logits");
}

}

#endif // LOSS_H