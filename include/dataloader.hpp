#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include "tensor.hpp"
using namespace linalg;

struct MNISTDataset {
    std::vector<Tensor<float>> images;
    std::vector<uint8_t> labels;

    bool load_images(const std::string& path);
    bool load_labels(const std::string& path);
    std::pair<Tensor<float>, uint8_t> get(size_t index) const;
};

#endif // DATALOADER_H