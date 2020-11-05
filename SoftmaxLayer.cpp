#include "SoftmaxLayer.h"

namespace DeepLearning {

SoftmaxLayer::SoftmaxLayer(unsigned int kernelSize)
    : Layer(LayerType::Softmax),
      m_kernelSize(kernelSize),
      m_loss(0){
    // TODO Auto-generated constructor stub

}

SoftmaxLayer::~SoftmaxLayer() {
    // TODO Auto-generated destructor stub
}

int SoftmaxLayer::Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        DBG_ERROR(("input is empty\n"));
        return -1;
    }

    if((1 != input.size()) || (input[0].rows() != this->m_kernelSize) || (input[0].cols() != 1)){
        DBG_ERROR(("input data error : (%lu, %lu, %lu)\n", input.size(), input[0].rows(), input[0].cols()));
        return -1;
    }

    // save input data.
    this->m_input = input;

    MatrixXf outputMatrix;
    this->InitOutputMatrix(outputMatrix);

    this->Softmax(input[0], outputMatrix);

    this->m_output.clear();
    this->m_output.push_back(outputMatrix);

    output = this->m_output;

    return 0;
}

int SoftmaxLayer::Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        DBG_ERROR(("input is empty\n"));
        return -1;
    }

    if((1 != input.size()) || (input[0].rows() != this->m_kernelSize) || (input[0].cols() != 1)){
        DBG_ERROR(("input data error : (%lu, %lu, %lu)\n", input.size(), input[0].rows(), input[0].cols()));
    }

    DSoftmax(input[0], this->m_input[0]);

    output = this->m_input;

    return 0;
}

int SoftmaxLayer::UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum){

    int ret = 0;

    if(isnan(m_loss)){
        DBG_ERROR(("loss is nan\n"));
        ret = -1;
    }

    //DBG_MSG(("%f", m_loss/batchSize));

    m_loss = 0;

    return ret;
}

int SoftmaxLayer::SaveLayerParameter(FILE *hFile){

    if(NULL != hFile){
        // layer type
        fprintf(hFile, "LayerType : %d\n", (int)this->m_layerType);
    }

    return 0;
}

int SoftmaxLayer::Softmax(const MatrixXf &inputMatrix, MatrixXf &outputMatrix){

    if((inputMatrix.rows() != this->m_kernelSize) || (inputMatrix.cols() != 1)){
        DBG_ERROR(("input matrix error : (%lu, %lu)\n", inputMatrix.rows(), inputMatrix.cols()));
        return -1;
    }

    // calculate max value in the matrix
    float max = inputMatrix(0, 0);
    for(unsigned int i = 0; i < inputMatrix.rows(); ++i){
        for(unsigned int j = 0; j < inputMatrix.cols(); ++j){
            if(inputMatrix(i, j) > max){
                max = inputMatrix(i, j);
            }
        }
    }

    // prevent exp over flow.
    MatrixXf input = inputMatrix;
    for(unsigned int i = 0; i < inputMatrix.rows(); ++i){
        for(unsigned int j = 0; j < inputMatrix.cols(); ++j){
            input(i, j) = input(i, j) - max;
        }
    }

    outputMatrix = input.array().exp() / input.array().exp().sum();

    return 0;
}

int SoftmaxLayer::DSoftmax(const MatrixXf &inputMatrix, MatrixXf &outputMatrix){

    if((inputMatrix.rows() != this->m_kernelSize) || (inputMatrix.rows() != outputMatrix.rows())){
        DBG_ERROR(("input data error : (%lu, %lu, %lu, %lu)\n", inputMatrix.rows(), inputMatrix.cols(), outputMatrix.rows(), outputMatrix.cols()));
        return -1;
    }

    outputMatrix = this->m_output[0] - inputMatrix;
    this->m_loss += outputMatrix.array().abs().sum();

    return 0;
}

int SoftmaxLayer::InitOutputMatrix(MatrixXf &outputMatrix){

    outputMatrix = MatrixXf::Zero(this->m_kernelSize, 1);

    return 0;
}

} /* namespace DeepLearning */
