#include "ReluLayer.h"

namespace DeepLearning {

ReluLayer::ReluLayer()
    : ActivationLayer(LayerType::ReLU){
    // TODO Auto-generated constructor stub

}

ReluLayer::~ReluLayer() {
    // TODO Auto-generated destructor stub
}


int ReluLayer::ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix){

    unsigned int m = input.rows();
    unsigned int n = input.cols();

    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputMatrix(i, j) = input(i, j) > 0 ? input(i, j) : 0;
        }
    }

    return 0;
}

int ReluLayer::BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad){

    unsigned int m = inputGrad.rows();
    unsigned int n = inputGrad.cols();

    // f(x)' = x > 0 ? 1 : 0
    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputGrad(i, j) = inputGrad(i, j) * (inputData(i, j) > 0 ? 1 : 0);
        }
    }

    return 0;
}

} /* namespace DeepLearning */
