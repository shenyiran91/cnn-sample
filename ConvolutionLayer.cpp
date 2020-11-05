#include "ConvolutionLayer.h"

namespace DeepLearning {

ConvolutionLayer::ConvolutionLayer(unsigned int kernelSize, unsigned int channleCount, unsigned int stride, MatrixXi linkMatrix, PadType padType)
    : Layer(LayerType::Convolution),
      m_kernelSize(kernelSize),
      m_channleCount(channleCount),
      m_stride(stride),
      m_linkMatrix(linkMatrix),
      m_padType(padType){

    // TODO Auto-generated constructor stub

    m_weight.clear();
    m_weight_gradient.clear();

    m_bias.clear();
    m_bias_gradient.clear();
}

ConvolutionLayer::~ConvolutionLayer() {
    // TODO Auto-generated destructor stub

    m_weight.clear();
    m_weight_gradient.clear();

    m_bias.clear();
    m_bias_gradient.clear();
}

/*-----------------------------------------------------------------------------------
 * public interface
 *---------------------------------------------------------------------------------*/
int ConvolutionLayer::Forward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    // save input data.
    this->m_input = input;

    // weight and bias matrix only initialize one time, and update in backward progress.
    InitWeightMatrix();
    InitBiasMatrix();

    // using current weight and bias calculate output.
    this->m_output.clear();
    for(unsigned int i = 0; i < this->m_channleCount; ++i){
        MatrixXf outputMatrix;
        this->InitOutputMatrix(input[0], outputMatrix);

        // calculate convolution
        for(unsigned int j = 0; j < input.size(); ++j){
            if(1 == this->m_linkMatrix(i, j)){
                MatrixXf tmpMatrix;
                this->InitOutputMatrix(input[j], tmpMatrix);
                this->Convolution(this->m_padType, input[j], this->m_weight[i][j], tmpMatrix);
                outputMatrix += tmpMatrix;
            }
        }

        // add bias
        MatrixXf bias = MatrixXf::Ones(outputMatrix.rows(), outputMatrix.cols());
        bias *= this->m_bias[i];
        outputMatrix += bias;

        // update output
        this->m_output.push_back(outputMatrix);
    }

    output = this->m_output;

    // save input and output to image
    //MatrixToImage(this->m_input, "con_input.jpg");
    //MatrixToImage(this->m_output, "con_output.jpg");

    return 0;
}

int ConvolutionLayer::Backward(const vector<MatrixXf> &input, vector<MatrixXf> &output){

    if(input.empty()){
        return -1;
    }

    // update weight
    for(unsigned int i = 0; i < this->m_input.size(); ++i){
        MatrixXf rotInput;
        Rotating180degree(this->m_input[i], rotInput);

        for(unsigned int j = 0; j < input.size(); ++j){
            if(1 == this->m_linkMatrix(j, i)){
                PadType padType;
                this->GetBackPadType(padType);

                MatrixXf weightMatrix = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                this->Convolution(padType, input[j], rotInput, weightMatrix);
                this->m_weight_gradient[j][i] += weightMatrix;
            }
        }

        this->m_input[i] = MatrixXf::Zero(this->m_input[i].rows(), this->m_input[i].cols());
    }

    // update bias
    for(unsigned int i = 0; i < input.size(); ++i){
        this->m_bias_gradient[i] += input[i].array().sum();
    }

    // back update input error
    for(unsigned int i = 0; i < input.size(); ++i){
        for(unsigned int j = 0; j < this->m_input.size(); ++j){
            if(1 == this->m_linkMatrix(i, j)){
                MatrixXf rotKernel;
                Rotating180degree(this->m_weight[i][j], rotKernel);

                PadType padType;
                this->GetBackPadType(padType);

                MatrixXf errorMatrix = MatrixXf::Zero(this->m_input[0].rows(), this->m_input[0].cols());
                this->Convolution(padType, input[i], rotKernel, errorMatrix);

                this->m_input[j] += errorMatrix;
            }
        }
    }

    output = this->m_input;

    return 0;
}

int ConvolutionLayer::UpdateGradient(unsigned int batchSize, float learningRate, float weightDecayRate, float momentum){

    for(unsigned int i = 0; i < this->m_weight.size(); ++i){
        for(unsigned int j = 0; j < this->m_weight[i].size(); ++j){
            // update weight gradient with old weight gradient
            this->m_weight_gradient[i][j] += this->m_old_weight_gradient[i][j] * momentum;

            // update weight matrix with weight gradient and weight decay
            this->m_weight[i][j] -= learningRate * this->m_weight_gradient[i][j] / batchSize + learningRate * weightDecayRate * this->m_weight[i][j];

            // save current weight gradient as old weight gradient.
            this->m_old_weight_gradient[i][j] = this->m_weight_gradient[i][j];

            // clear weight gradient matrix
            this->m_weight_gradient[i][j] = MatrixXf::Zero(this->m_weight_gradient[i][j].rows(), this->m_weight_gradient[i][j].cols());
        }
    }

    for(unsigned int i = 0; i < this->m_bias.size(); ++i){
        // update bias
        this->m_bias[i] -= (learningRate * this->m_bias_gradient[i] / batchSize);

        // clear bias gradient
        this->m_bias_gradient[i] = 0.0f;
    }

    return 0;
}

int ConvolutionLayer::SaveLayerParameter(FILE *hFile){

    if(NULL != hFile){
        // layer type
        fprintf(hFile, "LayerType : %d\n", (int)this->m_layerType);

        fprintf(hFile, "KernelSize : %u\n", this->m_kernelSize);
        fprintf(hFile, "ChannleCount : %u\n", this->m_channleCount);
        fprintf(hFile, "Stride : %u\n", this->m_stride);

        fprintf(hFile, "PadType : %d\n", (int)this->m_padType);

        // link matrix
        for(unsigned int i = 0; i < this->m_linkMatrix.rows(); ++i){
            for(unsigned int j = 0; j < this->m_linkMatrix.cols(); ++j){
                fprintf(hFile, "%d\t", this->m_linkMatrix(i, j));
            }
            fprintf(hFile, "\n");
        }

        // weight
        for(unsigned int i = 0; i < this->m_weight.size(); ++i){
            for(unsigned int j = 0; j < this->m_weight[i].size(); ++j){
                for(unsigned int m = 0; m < this->m_weight[i][j].rows(); ++m){
                    for(unsigned int n = 0; n < this->m_weight[i][j].cols(); ++n){
                        fprintf(hFile, "%f\t", this->m_weight[i][j](m, n));
                    }
                    fprintf(hFile, "\n");
                }

            }
        }

        // bias
        for(unsigned int i = 0; i < this->m_bias.size(); ++i){
            fprintf(hFile, "%f\t", this->m_bias[i]);
        }
        fprintf(hFile, "\n");
    }

    return 0;
}

/*-----------------------------------------------------------------------------------
 * private interface
 *----------------------------------------------------------------------------------*/
int ConvolutionLayer::Convolution(const PadType padType, const MatrixXf &input, const MatrixXf &conv, MatrixXf &output){

    // padding the input matrix
    MatrixXf padMatrix;
    this->PaddingMatrix(padType, input, padMatrix);

    // convolution compute
    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            output(i, j) = (padMatrix.block(i, j, conv.rows(), conv.cols()).array() * conv.array()).sum();
        }
    }

    return 0;
}


int ConvolutionLayer::InitWeightMatrix(){

    // only one time to initialize
    if(this->m_weight.empty()){
        for(unsigned int i = 0; i < this->m_channleCount; ++i){
            vector<MatrixXf> kernel;
            vector<MatrixXf> gradientKernel;
            vector<MatrixXf> oldGradientKernel;
            for(unsigned int j = 0; j < this->m_input.size(); ++j){
                MatrixXf weightMatrix = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                NormalDistribution(weightMatrix);

                for(unsigned int i = 0; i < weightMatrix.rows(); ++i){
                    for(unsigned int j = 0; j < weightMatrix.cols(); ++j){
                        weightMatrix(i, j) *= sqrtf(2.0f / (weightMatrix.rows() * weightMatrix.cols()));
                    }
                }

                kernel.push_back(weightMatrix);

                // init gradient weight matrix must be zero
                MatrixXf gradientMatrix = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                gradientKernel.push_back(gradientMatrix);

                // init old gradient weight matrix must be zero
                MatrixXf oldGradientMatrix = MatrixXf::Zero(this->m_kernelSize, this->m_kernelSize);
                oldGradientKernel.push_back(oldGradientMatrix);
            }

            this->m_weight.push_back(kernel);
            this->m_weight_gradient.push_back(gradientKernel);
            this->m_old_weight_gradient.push_back(oldGradientKernel);
        }
    }

    return 0;
}

int ConvolutionLayer::InitBiasMatrix(){

    // only one time to initialize
    if(this->m_bias.empty()){
        for(unsigned int i = 0; i < this->m_channleCount; ++i){
            MatrixXf biasMatrix = MatrixXf::Zero(1, 1);
            //NormalDistribution(biasMatrix);
            this->m_bias.push_back(biasMatrix(0, 0));

            // init gradient weight matrix must be zero
            this->m_bias_gradient.push_back(0.0f);
        }
    }

    return 0;
}


int ConvolutionLayer::InitOutputMatrix(const MatrixXf &input, MatrixXf &outputMatrix){

    switch(this->m_padType){
        case PadType::Full:
        {
            // [m * n] * [k * k] = [(m + k - 1) * (n + k - 1)]
            outputMatrix = MatrixXf::Zero(input.rows() + this->m_kernelSize - 1, input.cols() + this->m_kernelSize - 1);
            break;
        }
        case PadType::Same:
        {
            // [m * n] * [k * k] = [m * n]
            outputMatrix = MatrixXf::Zero(input.rows(), input.cols());
            break;
        }
        case PadType::Valid:
        {
            // [m * n] * [k * k] = [(m - k + 1) * (n - k + 1)]
            outputMatrix = MatrixXf::Zero(input.rows() - this->m_kernelSize + 1, input.cols() - this->m_kernelSize + 1);
            break;
        }
        default:
        {
            // default as full : [m * n] * [k * k] = [(m + k - 1) * (n + k - 1)]
            outputMatrix = MatrixXf::Zero(input.rows() + this->m_kernelSize - 1, input.cols() + this->m_kernelSize - 1);
            break;
        }
    }

    return 0;
}

int ConvolutionLayer::PaddingMatrix(const PadType padType, const MatrixXf &input, MatrixXf &padMatrix){

    switch(padType){
        case PadType::Full:
        {
            // [m * n] -> [(m + (k - 1) * 2) * (n + (k - 1) * 2)]
            padMatrix = MatrixXf::Zero(input.rows() + (this->m_kernelSize - 1) * 2, input.cols() + (this->m_kernelSize - 1) * 2);
            padMatrix.block(this->m_kernelSize - 1, this->m_kernelSize - 1, input.rows(), input.cols()) = input;
            break;
        }
        case PadType::Same:
        {
            // [m * n] -> [(m + (k - 1)) * (n + (k - 1))]
            padMatrix = MatrixXf::Zero(input.rows() + (this->m_kernelSize - 1), input.cols() + (this->m_kernelSize - 1));
            padMatrix.block((this->m_kernelSize - 1)/2, (this->m_kernelSize - 1)/2, input.rows(), input.cols()) = input;
            break;
        }
        case PadType::Valid:
        {
            // [m * n] -> [m * n]
            padMatrix = MatrixXf::Zero(input.rows(), input.cols());
            padMatrix.block(0, 0, input.rows(), input.cols()) = input;
            break;
        }
        default:
        {
            // [m * n] -> [(m + (k - 1) * 2) * (n + (k - 1) * 2)]
            padMatrix = MatrixXf::Zero(input.rows() + (this->m_kernelSize - 1) * 2, input.cols() + (this->m_kernelSize - 1) * 2);
            padMatrix.block(this->m_kernelSize - 1, this->m_kernelSize - 1, input.rows(), input.cols()) = input;
            break;
        }
    }

    return 0;
}

int ConvolutionLayer::GetBackPadType(PadType &padType){

    switch(this->m_padType){
        case PadType::Full:
        {
            padType = PadType::Valid;
            break;
        }
        case PadType::Same:
        {
            padType = PadType::Same;
            break;
        }
        case PadType::Valid:
        {
            padType = PadType::Full;
            break;
        }
        default:
        {
            padType = PadType::Valid;
            break;
        }
    }

    return 0;
}

int ConvolutionLayer::Rotating180degree(const MatrixXf &input, MatrixXf &output){

    output = MatrixXf::Zero(input.rows(), input.cols());
    for(unsigned int i = 0; i < input.rows(); ++i){
        for(unsigned int j = 0; j < input.cols(); ++j){
            output(i, j) = input(input.rows() -1 -i, input.cols() -1 -j);
        }
    }

    return 0;
}

} /* namespace DeepLearning */
