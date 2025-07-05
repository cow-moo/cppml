#include <iostream>
#include "dataloader.hpp"
#include "linalg.hpp"
#include "autodiff.hpp"
#include "module.hpp"
#include "solver.hpp"
#include "loss.hpp"
#include "timing.hpp"

using linalg::Tensor;
using linalg::Shape;
using autodiff::Expression;

int main() {
    backend::BackendGuard guard(backend::BackendType::Cuda);

    dataloader::MNISTDataset train;
    assert(train.load_images("../datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte"));
    std::cout << "train images loaded" << std::endl;
    assert(train.load_labels("../datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte"));
    std::cout << "train labels loaded" << std::endl;

    //dataloader::MNISTDataset::print_img(train.get(0).first);

    std::vector<Tensor<float>> y;
    for (uint8_t label : train.labels) {
        y.emplace_back(Tensor<float>::zeros({10}));
        y.back()[label] = 1.0f;
    }

    dataloader::DataLoader<float> dl(train.images, y, 256);

    dataloader::MNISTDataset test;
    assert(test.load_images("../datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"));
    assert(test.load_labels("../datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"));

    Tensor<float> xTest(Shape({test.images.size(), 784}));
    Tensor<size_t> yTest(Shape({test.images.size()}));

    for (size_t i = 0; i < test.images.size(); i++) {
        xTest[i] = test.images[i];
        yTest[i] = test.labels[i];
    }

    module::MLP<float> model({784, 128, 64, 10});
    auto weights = model.weights();

    solver::GradientDescent gd(weights, 0.1);

    std::cout << "Ready" << std::endl;

    // Tensor<size_t> yPred = model(xTest).value().argmax();
    // int correct = (yPred == yTest).astype<int>().sum();
    // std::cout << "Test accuracy: " << correct << " / " << test.images.size() << " = " << ((float)correct / test.images.size()) << std::endl;

    for (int epoch = 0; epoch < 1; epoch++) {
        // float totalLoss = 0.0f;
        // int cnt = 0;
        timing::Profiler::reset();
        for (auto&& [x, y] : dl) {
            Expression<float> logits, loss;
            {
                timing::ScopedProfiler timer("Forward");
                logits = model(x);
            }
            {
                timing::ScopedProfiler timer("Loss");
                loss = loss::cross_entropy_logits(logits, y);
                // totalLoss += loss.value();
                // cnt++;
            }
            {
                timing::ScopedProfiler timer("Backward");
                loss.backward();
            }
            {
                timing::ScopedProfiler timer("Step");
                gd.step();
                gd.zero_grad();
            }
            {
                timing::ScopedProfiler timer("Shuffle");
                dl.shuffle();
            }
        }
        timing::Profiler::report(false);
        //timing::Profiler::report(true);
        //std::cout << "Average epoch training loss: " << totalLoss / cnt << std::endl;

        Tensor<size_t> yPred = model(xTest).value().argmax();
        int correct = (yPred == yTest).astype<int>().sum();
        std::cout << "Test accuracy: " << correct << " / " << test.images.size() << " = " << ((float)correct / test.images.size()) << std::endl;
    }
}