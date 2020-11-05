#ifndef RELULAYER_H_
#define RELULAYER_H_

#include "Layer.h"
#include "ActivationLayer.h"

namespace DeepLearning {

class ReluLayer: public ActivationLayer {
public:
    ReluLayer();
    virtual ~ReluLayer();

private:
    int ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix);
    int BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad);
};

} /* namespace DeepLearning */

#endif /* RELULAYER_H_ */
