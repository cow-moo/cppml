#ifndef LAYER_H
#define LAYER_H

namespace layers {

class Layer {
public:
    
};

class Linear : public Layer {
public:
    Linear(int outputDim);

private:
    int inputDim;
    int outputDim;
};

class ReLU : public Layer {
public:
    ReLU();
};

} // namespace layers

#endif // LAYER_H