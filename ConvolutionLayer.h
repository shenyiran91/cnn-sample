#ifndef CONVOLUTIONLAYER_H_
#define CONVOLUTIONLAYER_H_

#include "Layer.h"

namespace DeepLearning {

enum class PadType{
    Full = 0,
    Same,
    Valid
};

class ConvolutionLayer: public Layer {
public:
    ConvolutionLayer(unsigned int kernelSize, unsigned int channleCount, unsigned int stride, MatrixXi linkMatrix, PadType padType);
    virtual ~ConvolutionLayer();

    int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum);
    int SaveLayerParameter(FILE *hFile);

private:
    int InitWeightMatrix();
    int InitBiasMatrix();

    int InitOutputMatrix(const MatrixXf &input, MatrixXf &outputMatrix);

    int GetBackPadType(PadType &padType);
    int PaddingMatrix(const PadType padType, const MatrixXf &input, MatrixXf &padMatrix);

    int Convolution(const PadType padType, const MatrixXf &input, const MatrixXf &conv, MatrixXf &output);

    int Rotating180degree(const MatrixXf &input, MatrixXf &output);

private:
    // convolution kernel size
    unsigned int m_kernelSize;
    // convolution kernel channels count
    unsigned int m_channleCount;
    // convolution move stride
    unsigned int m_stride;

    // link matrix. each convolution kernel corresponds to a set of output in the previous layer.
    // m : private layer image count
    // n : current layer convolution kernel
    // matrix size : [n * m]
    MatrixXi m_linkMatrix;

    // padding type
    PadType m_padType;

    // weight matrix list. each convolution kernel corresponds to a weight matrix.
    // m : output channel cout
    // n : input channel cout
    // k : kernel size
    // matrix size : [m * n * k * k]
    vector<vector<MatrixXf>> m_weight;
    vector<vector<MatrixXf>> m_weight_gradient;
    vector<vector<MatrixXf>> m_old_weight_gradient;

    // bias list. each convolution kernel corresponds to a bias.
    // list size : the convolution kernel count of current layer.
    vector<float> m_bias;
    vector<float> m_bias_gradient;
};

} /* namespace DeepLearning */

#endif /* CONVOLUTIONLAYER_H_ */
