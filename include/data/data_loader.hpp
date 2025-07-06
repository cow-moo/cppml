#ifndef DATA_DATA_LOADER_H
#define DATA_DATA_LOADER_H

#include <vector>
#include <numeric>
#include "linalg.hpp"
#include <concepts>

namespace data {

template <typename>
struct is_tensor : std::false_type {};

template <typename T>
struct is_tensor<linalg::Tensor<T>> : std::true_type {};

template <typename Tuple>
struct all_tuple_elements_are_tensors;

template <typename... Ts>
struct all_tuple_elements_are_tensors<std::tuple<Ts...>> 
    : std::conjunction<is_tensor<Ts>...> {};

template <typename T>
concept DatasetConcept = requires(T a, size_t i) {
    typename T::Sample;
    { a.size() } -> std::convertible_to<size_t>;
    { a.get(i) } -> std::same_as<typename T::Sample>;
} && all_tuple_elements_are_tensors<typename T::Sample>::value;

template <DatasetConcept Dataset>
class DataLoader {
public:
    struct Iterator {
        const DataLoader* parent;
        size_t batchIdx;

        Iterator(const DataLoader* p, size_t batchIdx = 0) : 
            parent(p), batchIdx(batchIdx) {}

        template <typename T>
        auto get_batched_tensor(linalg::Shape shape) const {
            shape.insert(shape.begin(), parent->batchSize_);
            return linalg::Tensor<T>(shape, parent->backend_);
        }

        template <typename... Ts, std::size_t... Is>
        auto collate_impl(const std::vector<std::tuple<linalg::Tensor<Ts>...>>& batch, 
                          std::index_sequence<Is...>) const {
            auto res = std::make_tuple(get_batched_tensor<Ts>(std::get<Is>(batch[0]).shape())...);
            
            for (size_t i = 0; i < batch.size(); ++i) {
                (..., (std::get<Is>(res)[i] = std::get<Is>(batch[i]).to(parent->backend_)));
            }

            return res;
        }

        template <typename... Ts>
        auto collate(const std::vector<std::tuple<linalg::Tensor<Ts>...>>& batch) const {
            static_assert(sizeof...(Ts) > 0, "Tuples must be non-empty");
            return collate_impl(batch, std::index_sequence_for<Ts...>{});
        }

        Dataset::Sample operator*() const {
            std::vector<typename Dataset::Sample> samples;
            for (size_t i = 0; i < parent->batchSize_; ++i) {
                size_t shuffledIdx = parent->permutation_[batchIdx * parent->batchSize_ + i];
                samples.emplace_back(parent->dataset_.get(shuffledIdx));
            }
            return collate(samples);
        }

        Iterator& operator++() {
            ++batchIdx;
            return *this;
        }

        bool operator!=(const Iterator& other) const { return batchIdx != other.batchIdx; }
    };

    DataLoader(Dataset&& dataset, size_t batchSize, backend::BackendType backend = backend::current_backend_type) : 
        dataset_(std::move(dataset)), batchSize_(batchSize), permutation_(dataset_.size()), backend_(backend)
    {
        std::iota(permutation_.begin(), permutation_.end(), 0);
    }

    Iterator begin() const {
        return Iterator(this);
    }

    // Only does full batches
    Iterator end() const {
        return Iterator(this, dataset_.size() / batchSize_);
    }

    void shuffle() {
        std::shuffle(permutation_.begin(), permutation_.end(), rng_);
    }
private:
    Dataset dataset_;
    size_t batchSize_;
    std::vector<size_t> permutation_;
    mutable std::mt19937 rng_;
    backend::BackendType backend_;
};

}

#endif // DATA_DATALOADER_H