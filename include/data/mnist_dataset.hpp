#ifndef DATA_MNIST_DATASET_H
#define DATA_MNIST_DATASET_H

#include <utility>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include "linalg.hpp"

namespace data {

struct MNISTDataset {
    using Sample = std::tuple<linalg::Tensor<float>, linalg::Tensor<uint8_t>>;

    MNISTDataset(const std::string& imagesPath, const std::string& labelsPath);

    Sample get(size_t index) const;
    size_t size() const;

    linalg::Tensor<float> images;
    linalg::Tensor<uint8_t> labels;
    
    static void print_img(const linalg::Tensor<float>& img);
};

}

#endif // DATA_MNIST_DATASET_H