#include "dataloader.hpp"
#include <fstream>
#include "shape.hpp"

namespace dataloader {

static uint32_t read_uint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

bool MNISTDataset::load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    uint32_t magic = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    assert(magic == 2051 && rows == 28 && cols == 28);

    std::vector<uint8_t> flat(28 * 28);
    images.reserve(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(flat.data()), 28 * 28);
        images.emplace_back(Tensor<uint8_t>(flat).astype<float>() / 255.0f);
    }

    return true;
}

bool MNISTDataset::load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    uint32_t magic = read_uint32(file);
    uint32_t num_labels = read_uint32(file);
    assert(magic == 2049);

    labels.resize(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return true;
}

std::pair<Tensor<float>, uint8_t> MNISTDataset::get(size_t index) const {
    return {images[index], labels[index]};
}

void MNISTDataset::print_img(const Tensor<float>& img) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (img[i * 28 + j] < 0.5f ? '.' : '@') << ' ';
        }
        std::cout << std::endl;
    }
}

}