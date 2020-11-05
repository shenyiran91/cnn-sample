#ifndef FULLCONNECT_H_
#define FULLCONNECT_H_

#include "Layer.h"

namespace DeepLearning {

class FullConnect: public Layer {
public:
    FullConnect(unsigned int kernelSize);
    virtual ~FullConnect();

    int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum);
    int SaveLayerParameter(FILE *hFile);

private:
    int InitWeightMatrix(const MatrixXf input);
    int InitBiasMatrix();

    int InitOutputMatrix(MatrixXf &outputMatrix);

    int CombinMatrix(const vector<MatrixXf> &input, MatrixXf &output);
    int SplitMatrix(const MatrixXf &input, vector<MatrixXf> &output);

    int DebugUpdateGradient();

private:
    // kernel size
    unsigned int m_kernelSize;

    // save the input matrix vector to [n * 1] matrix.
    MatrixXf m_combined_input;

    // weight matrix for current layer
    // m : private layer neuron count
    // n : current layer neuron count
    // matrix size : [n * m]
    MatrixXf m_weight;
    MatrixXf m_weight_gradient;
    MatrixXf m_old_weight_gradient;

    // bias matrix list
    // matrix size : [n * 1]
    MatrixXf m_bias;
    MatrixXf m_bias_gradient;
};

} /* namespace DeepLearning */

#endif /* FULLCONNECT_H_ */
