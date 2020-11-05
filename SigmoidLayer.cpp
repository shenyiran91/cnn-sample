#include "SigmoidLayer.h"

namespace DeepLearning {

SigmoidLayer::SigmoidLayer()
    : ActivationLayer(LayerType::Sigmoid){
    // TODO Auto-generated constructor stub

}

SigmoidLayer::~SigmoidLayer() {
    // TODO Auto-generated destructor stub
}

int SigmoidLayer::ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix){

    unsigned int m = input.rows();
    unsigned int n = input.cols();

    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputMatrix(i, j) = 1.0f / (1 + exp(-input(i, j)));
        }
    }

    return 0;
}

int SigmoidLayer::BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad){

    unsigned int m = inputGrad.rows();
    unsigned int n = inputGrad.cols();

    // f(x)' = f(x) * (1 - f(x))
    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputGrad(i, j) = inputGrad(i, j) * (outputData(i, j) * (1.0f - outputData(i, j)));
        }
    }

    return 0;
}

} /* namespace DeepLearning */
