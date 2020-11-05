#include "PoolingLayer.h"

namespace DeepLearning {

PoolingLayer::PoolingLayer(unsigned int kernelSize, PoolType poolType)
    : Layer(LayerType::Pooling),
      m_kernelSize(kernelSize),
      m_poolType(poolType){
    // TODO Auto-generated constructor stub

}

PoolingLayer::~PoolingLayer() {
    // TODO Auto-generated destructor stub
}

int PoolingLayer::Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    // save input data.
    this->m_input = input;

    this->m_output.clear();
    for(unsigned int i = 0; i < input.size(); ++i){
        MatrixXf outputMatrix;
        InitOutputMatrix(input[i], outputMatrix);
        Pool(input[i], outputMatrix);
        this->m_output.push_back(outputMatrix);
    }

    output = this->m_output;

    // save input and output to image
    //MatrixToImage(this->m_input, "pool_input.jpg");
    //MatrixToImage(this->m_output, "pool_output.jpg");

    return 0;
}

int PoolingLayer::Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    output.clear();
    for(unsigned int i = 0; i < input.size(); ++i){
        DPool(input[i], this->m_input[i]);
    }

    output = this->m_input;

    return 0;
}

int PoolingLayer::UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum){

    // do nothing

    return 0;
}

int PoolingLayer::SaveLayerParameter(FILE *hFile){

    if(NULL != hFile){
        // layer type
        fprintf(hFile, "LayerType : %d\n", (int)this->m_layerType);

        fprintf(hFile, "KernelSize : %u\n", this->m_kernelSize);
        fprintf(hFile, "PoolType : %d\n", (int)this->m_poolType);
    }

    return 0;
}

int PoolingLayer::Pool(const MatrixXf &input, MatrixXf &output){

    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            if(PoolType::Max == this->m_poolType){
                MatrixXf tmp = input.block(this->m_kernelSize * i, this->m_kernelSize * j, this->m_kernelSize, this->m_kernelSize);
                unsigned maxI = 0, maxJ = 0;
                GetMaxItem(tmp, maxI, maxJ);
                output(i, j) = tmp(maxI, maxJ);
            }
            else if(PoolType::Mean == this->m_poolType){
                output(i, j) = input.block(this->m_kernelSize * i, this->m_kernelSize * j, this->m_kernelSize, this->m_kernelSize).sum() / (this->m_kernelSize * this->m_kernelSize);
            }
        }
    }

    return 0;
}

int PoolingLayer::DPool(const MatrixXf &input, MatrixXf &output){

    for (int i = 0; i < input.rows(); i++)
    {
        for (int j = 0; j < input.cols(); j++)
        {
            if(PoolType::Max == this->m_poolType){
                // calculate max item of current pooling
                MatrixXf tmp = output.block(this->m_kernelSize * i, this->m_kernelSize * j, this->m_kernelSize, this->m_kernelSize);
                unsigned maxI = 0, maxJ = 0;
                GetMaxItem(tmp, maxI, maxJ);
                output.block(this->m_kernelSize * i, this->m_kernelSize * j, this->m_kernelSize, this->m_kernelSize) = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                output(this->m_kernelSize * i + maxI, this->m_kernelSize * j + maxJ) = input(i, j);
            }
            else if(PoolType::Mean == this->m_poolType){
                MatrixXf outputMatrix = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                outputMatrix.fill(input(i, j) / (this->m_kernelSize * this->m_kernelSize));
                output.block(this->m_kernelSize * i, this->m_kernelSize * j, this->m_kernelSize, this->m_kernelSize) = outputMatrix;
            }
        }
    }

    return 0;
}

int PoolingLayer::InitOutputMatrix(const MatrixXf &input, MatrixXf &output){

    output = MatrixXf::Zero(input.rows() / this->m_kernelSize, input.cols() / this->m_kernelSize);

    return 0;
}

int PoolingLayer::GetMaxItem(const MatrixXf &input, unsigned int &maxI, unsigned int &maxJ){

    float value = input(0, 0);

    maxI = 0, maxJ = 0;
    for(unsigned int i = 0; i < input.rows(); ++i){
        for(unsigned int j = 0; j < input.cols(); ++j){
            if(input(i, j) > value){
                value = input(i, j);
                maxI = i;
                maxJ = j;
            }
        }
    }

    return 0;
}

} /* namespace DeepLearning */
