#ifndef TANHLAYER_H_
#define TANHLAYER_H_

#include "ActivationLayer.h"

namespace DeepLearning {

class TanhLayer: public ActivationLayer {
public:
    TanhLayer();
    virtual ~TanhLayer();

private:
    int ForwardActivation(const MatrixXf &input, MatrixXf &outputMatrix);
    int BackwardActivation(const MatrixXf &inputData, const MatrixXf &outputData, const MatrixXf &inputGrad, MatrixXf &outputGrad);
};

} /* namespace DeepLearning */

#endif /* TANHLAYER_H_ */
