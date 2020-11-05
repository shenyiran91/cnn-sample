#include "ActivationLayer.h"

namespace DeepLearning {

ActivationLayer::ActivationLayer(LayerType layerType)
    : Layer(layerType){
    // TODO Auto-generated constructor stub

}

ActivationLayer::~ActivationLayer() {
    // TODO Auto-generated destructor stub
}

/*-----------------------------------------------------------------------------------
 * public interface
 *----------------------------------------------------------------------------------*/
int ActivationLayer::Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    // save input data.
    this->m_input = input;

    this->m_output.clear();
    for(unsigned int i = 0; i < input.size(); ++i){

        MatrixXf outputMatrix;
        InitOutputMatrix(input[i], outputMatrix);
        ForwardActivation(input[i], outputMatrix);

        this->m_output.push_back(outputMatrix);
    }

    output = this->m_output;

    return 0;
}

int ActivationLayer::Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    output.clear();
    for(unsigned int i = 0; i < input.size(); ++i){
        // input matrix vector is equal to space of output matrix vector
        BackwardActivation(this->m_input[i], this->m_output[i], input[i], this->m_input[i]);
    }

    output = this->m_input;

    return 0;
}

int ActivationLayer::UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum){

    // do nothing

    return 0;
}

int ActivationLayer::SaveLayerParameter(FILE *hFile){

    if(NULL != hFile){
        // only layer type, activation layer no weight parameters.
        fprintf(hFile, "LayerType : %d\n", (int)this->m_layerType);
    }

    return 0;
}

/*-----------------------------------------------------------------------------------
 * private interface
 *----------------------------------------------------------------------------------*/
int ActivationLayer::InitOutputMatrix(const MatrixXf &input, MatrixXf &output){

    output = MatrixXf::Zero(input.rows(), input.cols());

    return 0;
}

} /* namespace DeepLearning */
