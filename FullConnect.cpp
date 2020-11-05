#include "FullConnect.h"

namespace DeepLearning {

FullConnect::FullConnect(unsigned int kernelSize)
    : Layer(LayerType::FullConnect),
      m_kernelSize(kernelSize){
    // TODO Auto-generated constructor stub

}

FullConnect::~FullConnect() {
    // TODO Auto-generated destructor stub

}

/*-----------------------------------------------------------------------------------
 * public interface
 *----------------------------------------------------------------------------------*/
int FullConnect::Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    // save input data.
    this->m_input = input;

    // change input matrix vector to [n * 1] matrix.
    CombinMatrix(this->m_input, this->m_combined_input);

    // initialize weight and bias matrix only one time.
    InitWeightMatrix(this->m_combined_input);
    InitBiasMatrix();

    MatrixXf outputMatrix;
    this->InitOutputMatrix(outputMatrix);

    // using current weight and bias calculate output.
    // [n * m] * [m * 1] = [n * 1]
    outputMatrix = this->m_weight * this->m_combined_input + this->m_bias;

    this->m_output.clear();
    this->m_output.push_back(outputMatrix);

    output = this->m_output;

    return 0;
}

int FullConnect::Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    MatrixXf diffMatrix = input[0];

    // update weight
    // [n * 1] * [1 * m] = [n * m]
    this->m_weight_gradient += (diffMatrix * this->m_combined_input.transpose());

    // update bias
    this->m_bias_gradient += diffMatrix;

    // back update input
    // [m * n] * [n * 1] = [m * 1]
    this->m_combined_input = this->m_weight.transpose() * diffMatrix;

    // convert [m * 1] matrix to input list
    SplitMatrix(this->m_combined_input, this->m_input);

    output = this->m_input;

    return 0;
}

int FullConnect::UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum){

    // update weight gradient with old weight gradient
    this->m_weight_gradient += this->m_old_weight_gradient * momentum;

    // update weight matrix
    this->m_weight -= learningRate * this->m_weight_gradient / batchSize + learningRate * weightDecayRate * this->m_weight;

    // save current weight gradient as old weight gradient.
    this->m_old_weight_gradient = this->m_weight_gradient;

    // clear weight gradient matrix
    this->m_weight_gradient = MatrixXf::Zero(this->m_weight_gradient.rows(), this->m_weight_gradient.cols());

    // update bias
    this->m_bias -= (learningRate * this->m_bias_gradient / batchSize);
    // clear bias gradient
    this->m_bias_gradient = MatrixXf::Zero(this->m_bias_gradient.rows(), this->m_bias_gradient.cols());

    return 0;
}

int FullConnect::SaveLayerParameter(FILE *hFile){

    if(NULL != hFile){
        fprintf(hFile, "LayerType : %d\n", (int)this->m_layerType);
        fprintf(hFile, "KernelSize : %u\n", this->m_kernelSize);

        // weight
        for(unsigned int i = 0; i < this->m_weight.rows(); ++i){
            for(unsigned int j = 0; j < this->m_weight.cols(); ++j){
                fprintf(hFile, "%f\t", this->m_weight(i, j));
            }
            fprintf(hFile, "\n");
        }

        // bias
        for(unsigned int i = 0; i < this->m_bias.rows(); ++i){
            for(unsigned int j = 0; j < this->m_bias.cols(); ++j){
                fprintf(hFile, "%f\t", this->m_bias(i, j));
            }
            fprintf(hFile, "\n");
        }
    }

    return 0;
}

/*-----------------------------------------------------------------------------------
 * private interface
 *----------------------------------------------------------------------------------*/
int FullConnect::InitWeightMatrix(const MatrixXf input){

    if((0 == this->m_weight.rows()) || (0 == this->m_weight.cols())){
        this->m_weight = MatrixXf::Zero(this->m_kernelSize, input.rows());
        NormalDistribution(this->m_weight);

        for(unsigned int i = 0; i < this->m_weight.rows(); ++i){
            for(unsigned int j = 0; j < this->m_weight.cols(); ++j){
                this->m_weight(i, j) /= sqrtf(this->m_weight.rows() * this->m_weight.cols());
            }
        }

        this->m_weight_gradient = MatrixXf::Zero(this->m_kernelSize, input.rows());
        this->m_old_weight_gradient = MatrixXf::Zero(this->m_kernelSize, input.rows());
    }

    return 0;
}

int FullConnect::InitBiasMatrix(){

    if((0 == this->m_bias.rows()) || (0 == this->m_bias.cols())){
        this->m_bias = MatrixXf::Zero(this->m_kernelSize, 1);
        //NormalDistribution(this->m_bias);
        for(unsigned int i = 0; i < this->m_bias.rows(); ++i){
            for(unsigned int j = 0; j < this->m_bias.cols(); ++j){
                this->m_bias(i, j) /= sqrtf(this->m_bias.rows() * this->m_bias.cols());
            }
        }

        this->m_bias_gradient = MatrixXf::Zero(this->m_kernelSize, 1);
    }

    return 0;
}

int FullConnect::InitOutputMatrix(MatrixXf &outputMatrix){

    outputMatrix = MatrixXf::Zero(this->m_kernelSize, 1);

    return 0;
}

int FullConnect::CombinMatrix(const vector<MatrixXf> &input, MatrixXf &output){

    // change input matrix vector to [n * 1] matrix.
    int count = 0;
    for(unsigned int i = 0; i < input.size(); ++i){
        count += input[i].rows() * input[i].cols();
    }

    if((output.rows() != count) || (output.cols() != 1)){
        output = MatrixXf::Zero(count, 1);
    }

    for(unsigned int i = 0; i < input.size(); ++i){
        for(unsigned int j = 0; j < input[i].rows(); ++j){
            for(unsigned int k = 0; k < input[i].cols(); ++k){
            	output(i * (input[i].rows() * input[i].cols()) + j * input[i].cols() + k, 0) = input[i](j, k);
            }
        }
    }

    return 0;
}

int FullConnect::SplitMatrix(const MatrixXf &input, vector<MatrixXf> &output){

    // convert [m * 1] matrix to input list
    for(unsigned int i = 0; i < output.size(); ++i){
        for(unsigned int j = 0; j < output[i].rows(); ++j){
            for(unsigned int k = 0; k < output[i].cols(); ++k){
            	output[i](j, k) = input(i * (this->m_input[i].rows() * output[i].cols()) + j * output[i].cols() + k, 0);
            }
        }
    }

    return 0;
}

/*-----------------------------------------------------------------------------------
 * for debug interface
 *----------------------------------------------------------------------------------*/
int FullConnect::DebugUpdateGradient(){
    for(unsigned int i = 0; i < this->m_weight.rows(); ++i){
        for(unsigned int j = 0; j < this->m_weight.cols(); ++j){
            if(this->m_weight(i, j) > 1.0){
                printf("m_weight(%u, %u) %f\n", i, j, this->m_weight(i, j));
            }
        }
    }

    for(unsigned int i = 0; i < this->m_weight_gradient.rows(); ++i){
        for(unsigned int j = 0; j < this->m_weight_gradient.cols(); ++j){
            if(this->m_weight_gradient(i, j) > 100){
                printf("m_weight_gradient(%u, %u) %f, %f\n", i, j, this->m_weight(i, j), this->m_weight_gradient(i, j));
            }
        }
    }

    for(unsigned int i = 0; i < this->m_bias.rows(); ++i){
        for(unsigned int j = 0; j < this->m_bias.cols(); ++j){
            if(this->m_bias(i, j) > 10){
                printf("(m_bias(%u, %u) %f\n", i, j, this->m_bias(i, j));
            }
        }
    }

    for(unsigned int i = 0; i < this->m_bias_gradient.rows(); ++i){
        for(unsigned int j = 0; j < this->m_bias_gradient.cols(); ++j){
            if(this->m_bias_gradient(i, j) > 100){
                printf("m_bias_gradient(%u, %u) %f\n", i, j, this->m_bias_gradient(i, j));
            }
        }
    }

    return 0;
}

} /* namespace DeepLearning */
