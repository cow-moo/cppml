#include "data/mnist_dataset.hpp"
#include "timing.hpp"
#include <fstream>
#include <cassert>
#include "linalg.hpp"

namespace data {

static constexpr backend::BackendType backend = backend::BackendType::CpuSingleThread;

static uint32_t read_uint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

static std::vector<uint8_t> get_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::ios_base::failure("failed to open images file");

    uint32_t magic = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);
    assert(magic == 2051 && rows == 28 && cols == 28);

    std::vector<uint8_t> res(num_images * 28 * 28);
    file.read(reinterpret_cast<char*>(res.data()), num_images * 28 * 28);
    return res;
}

static std::vector<uint8_t> get_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::ios_base::failure("failed to open labels file");

    uint32_t magic = read_uint32(file);
    uint32_t num_labels = read_uint32(file);
    assert(magic == 2049);

    std::vector<uint8_t> res(num_labels);
    file.read(reinterpret_cast<char*>(res.data()), num_labels);
    return res;
}

MNISTDataset::MNISTDataset(const std::string& imagesPath, const std::string& labelsPath) :
    images(linalg::Tensor<uint8_t>(get_images(imagesPath), backend).astype<float>() / 255.0f),
    labels(get_labels(labelsPath), backend)
{
    images.assign(images.reshape({size(), 28 * 28}));
}

MNISTDataset::Sample MNISTDataset::get(size_t index) const {
    return {images[index], labels[index]};
}

size_t MNISTDataset::size() const {
    return labels.shape()[0];
}

void MNISTDataset::print_img(const linalg::Tensor<float>& img) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (img[i * 28 + j] < 0.5f ? '.' : '@') << ' ';
        }
        std::cout << std::endl;
    }
}

}