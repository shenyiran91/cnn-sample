#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include "Layer.h"

namespace DeepLearning {

class ActivationLayer: public Layer {
public:
    ActivationLayer(LayerType layerType);
    virtual ~ActivationLayer();

    int Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output);
    int UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum);
    int SaveLayerParameter(FILE *hFile);

private:
    int InitOutputMatrix(const MatrixXf &input, MatrixXf &outputMatrix);

protected:
    virtual int ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix) = 0;
    virtual int BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad) = 0;
};

} /* namespace DeepLearning */

#endif /* ACTIVATIONLAYER_H_ */
