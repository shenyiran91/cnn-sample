#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include "Layer.h"

namespace DeepLearning {

class SoftmaxLayer: public Layer {
public:
    SoftmaxLayer(unsigned int kernelSize);
    virtual ~SoftmaxLayer();

    int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum);

    int SaveLayerParameter(FILE *hFile);
    int SaveLayerInputOutput(FILE *hFile);

private:
    int InitOutputMatrix(MatrixXf &outputMatrix);

    int Softmax(const MatrixXf &inputMatrix, MatrixXf &outputMatrix);
    int DSoftmax(const MatrixXf &inputMatrix, MatrixXf &outputMatrix);

    int DebugInputOutput();

private:
    // kernel size
    unsigned int m_kernelSize;

    // weight matrix for current layer
    // m : private layer neuron count
    // n : current layer neuron count
    // matrix size : [n * m]
    MatrixXf m_weight;

    double m_loss;
};

} /* namespace DeepLearning */

#endif /* SOFTMAXLAYER_H_ */
