#ifndef POOLINGLAYER_H_
#define POOLINGLAYER_H_

#include "Layer.h"

namespace DeepLearning {

enum class PoolType{
    Max,
    Mean,
};

class PoolingLayer: public Layer {
public:
    PoolingLayer(unsigned int kernelSize, PoolType poolType);
    virtual ~PoolingLayer();

    int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum);
    int SaveLayerParameter(FILE *hFile);

private:
    int InitOutputMatrix(const MatrixXf &input, MatrixXf &outputMatrix);

    int Pool(const MatrixXf &input, MatrixXf &output);
    int DPool(const MatrixXf &input, MatrixXf &output);

    int GetMaxItem(const MatrixXf &input, unsigned int &maxI, unsigned int &maxJ);

private:
    unsigned int m_kernelSize;
    PoolType m_poolType;
};

} /* namespace DeepLearning */

#endif /* POOLINGLAYER_H_ */
