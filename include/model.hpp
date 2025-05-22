#ifndef MODEL_H
#define MODEL_H

#include "layer.hpp"
#include "solver.hpp"
#include <vector>

namespace models {

class Model {
public:
    virtual void fit();
};

class Sequential : public Model {
public:
    Sequential(std::size_t inputDim, std::vector<layers::Layer> layers, solver::Optimizer optimizer, solver::LossFunction loss);
};

} // namespace model

#endif // MODEL_H