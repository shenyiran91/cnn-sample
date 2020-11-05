#include "TanhLayer.h"

namespace DeepLearning {

TanhLayer::TanhLayer()
: ActivationLayer(LayerType::Tanh){
    // TODO Auto-generated constructor stub

}

TanhLayer::~TanhLayer() {
    // TODO Auto-generated destructor stub
}

int TanhLayer::ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix){

    unsigned int m = input.rows();
    unsigned int n = input.cols();

    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputMatrix(i, j) = tanh(input(i, j));
        }
    }

    return 0;
}

int TanhLayer::BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad){

    unsigned int m = inputGrad.rows();
    unsigned int n = inputGrad.cols();

    // f(x)' = 1 - pow(f(x), 2)
    for(unsigned int i = 0; i < m; ++i){
        for(unsigned int j = 0; j < n; ++j){
            outputGrad(i, j) = inputGrad(i, j) * (1.0f - pow(outputData(i, j), 2));
        }
    }

    return 0;
}

} /* namespace DeepLearning */
