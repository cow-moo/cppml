#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <numeric>
#include "linalg.hpp"

namespace dataloader {

using linalg::Tensor;
using linalg::Shape;

template <typename T = float>
class DataLoader {
public:
    struct Iterator {
        // Could make this cache results and reuse memory location
        const DataLoader* parent;
        size_t batchIdx;

        Iterator(const DataLoader* p, size_t batchIdx = 0) 
            : parent(p), batchIdx(batchIdx) {

        }

        std::pair<Tensor<T>, Tensor<T>> operator*() const {
            Tensor<T> xBatch(parent->batchShapes_.first);
            Tensor<T> yBatch(parent->batchShapes_.second);
            //std::pair<Tensor<T>, Tensor<T>> batch;
            for (size_t i = 0; i < parent->batchSize_; i++) {
                size_t shuffledIdx = parent->permutation_[batchIdx * parent->batchSize_ + i];
                xBatch[i] = parent->x_[shuffledIdx];
                yBatch[i] = parent->y_[shuffledIdx];
            }
            
            return {xBatch, yBatch};
        }

        Iterator& operator++() {
            batchIdx++;
            return *this;
        }

        bool operator!=(const Iterator& other) const { return batchIdx != other.batchIdx; }
    };

    DataLoader(const std::vector<Tensor<T>> x, const std::vector<Tensor<T>> y, size_t batchSize)
        : x_(x), y_(y), batchSize_(batchSize), permutation_(x.size()) {
        assert(x.size() == y.size());
        std::iota(permutation_.begin(), permutation_.end(), 0);

        Shape xShape = x[0].shape();
        xShape.insert(xShape.begin(), batchSize);
        Shape yShape = y[0].shape();
        yShape.insert(yShape.begin(), batchSize);
        batchShapes_ = {xShape, yShape};
    }

    Iterator begin() const {
        return Iterator(this, 0);
    }

    Iterator end() const {
        return Iterator(this, x_.size() / batchSize_);
    }

    void shuffle() {
        std::shuffle(permutation_.begin(), permutation_.end(), rng_);
    }

private:
    std::vector<Tensor<T>> x_, y_;
    size_t batchSize_;
    std::pair<Shape, Shape> batchShapes_;
    std::vector<size_t> permutation_;
    mutable std::mt19937 rng_;
};

struct MNISTDataset {
    template <typename T>
    using Tensor = linalg::Tensor<T>;
    
    std::vector<Tensor<float>> images;
    std::vector<uint8_t> labels;

    bool load_images(const std::string& path);
    bool load_labels(const std::string& path);
    std::pair<Tensor<float>, uint8_t> get(size_t index) const;
    
    static void print_img(const Tensor<float>& img);
};

}

#endif // DATALOADER_H