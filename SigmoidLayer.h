#ifndef SIGMOIDLAYER_H_
#define SIGMOIDLAYER_H_

#include "ActivationLayer.h"

namespace DeepLearning {

class SigmoidLayer: public ActivationLayer {
public:
    SigmoidLayer();
    virtual ~SigmoidLayer();

private:
    int ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix);
    int BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad);
};

} /* namespace DeepLearning */

#endif /* SIGMOIDLAYER_H_ */
