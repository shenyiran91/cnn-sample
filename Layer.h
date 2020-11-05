#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <Eigen/Dense>

#include "DebugPrint.h"

using namespace std;
using namespace Eigen;

namespace DeepLearning {

enum class LayerType{
    Convolution,
    Pooling,
    FullConnect,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

class Layer {
public:
    Layer(LayerType layerType);
    virtual ~Layer();

    virtual int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output) = 0;
    virtual int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output) = 0;
    virtual int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum) = 0;
    virtual int SaveLayerParameter(FILE *hFile) = 0;

protected:
    int NormalDistribution(MatrixXf &matrix);
    int UniformDistribution(MatrixXf &matrix);

    int MatrixToImage(const vector<MatrixXf> &matrix, string path);

protected:
    LayerType m_layerType;
    vector<MatrixXf> m_input;
    vector<MatrixXf> m_output;
};

} /* namespace DeepLearning */

#endif /* LAYER_H_ */
