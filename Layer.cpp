#include "Layer.h"

#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

namespace DeepLearning {

Layer::Layer(LayerType layerType)
    : m_layerType(layerType){
    // TODO Auto-generated constructor stub

    this->m_input.clear();
    this->m_output.clear();
}

Layer::~Layer() {
    // TODO Auto-generated destructor stub

    this->m_input.clear();
    this->m_output.clear();
}

/*----------------------------------------------------------------------------
 * about random:
 * 1. the same seed create the same random sequence in different call.
 * 2. static random engine create different sequence in different call.
 * 3. no static random engine with time as seed maybe create same random
 *    sequence when time interval is too short.
 *---------------------------------------------------------------------------*/
int Layer::NormalDistribution(MatrixXf &matrix){

    // initialize weight matrix using normal distribution.
    //static default_random_engine randomEngine(time(NULL));
    //static normal_distribution<float> normalDistribution(0.0f, 1.0f);

    static default_random_engine randomEngine(100);
    static normal_distribution<float> normalDistribution(0.0f, 1.0f);

    for(unsigned int i = 0; i < matrix.rows(); ++i){
        for(unsigned int j = 0; j < matrix.cols(); ++j){
            matrix(i, j) = normalDistribution(randomEngine);
        }
    }

    return 0;
}

int Layer::UniformDistribution(MatrixXf &matrix){

    // initialize weight matrix using normal distribution.
    static default_random_engine randomEngine(time(NULL));
    static uniform_real_distribution<float> normalDistribution(-0.1f, 0.1f);
    for(unsigned int i = 0; i < matrix.rows(); ++i){
        for(unsigned int j = 0; j < matrix.cols(); ++j){
            matrix(i, j) = normalDistribution(randomEngine);
        }
    }

    return 0;
}

int Layer::MatrixToImage(const vector<MatrixXf> &matrix, string path){

    unsigned int w = 0, h = 0;
    for(unsigned int i = 0; i < matrix.size(); ++i){
        if(matrix[i].rows() > h){
            h = matrix[i].rows();
        }

        w += matrix[i].cols();
    }

    Mat image(h, w, CV_8UC1);

    unsigned int curCol = 0;
    for(unsigned int i = 0; i < matrix.size(); ++i){
        float maxValue = matrix[i](0, 0);
        float minValue = matrix[i](0, 0);
        for(unsigned int j = 0; j < matrix[i].rows(); ++j){
            for(unsigned int k = 0; k < matrix[i].cols(); ++k){
                if(matrix[i](j, k) > maxValue){
                    maxValue = matrix[i](j, k);
                }
                else if(matrix[i](j, k) < minValue){
                    minValue = matrix[i](j, k);
                }
            }
        }

        for(unsigned int j = 0; j < matrix[i].rows(); ++j){
            for(unsigned int k = 0; k < matrix[i].cols(); ++k){
                image.at<uchar>(j, curCol + k) = (uchar) (255 * (matrix[i](j, k) - minValue)  / (maxValue - minValue));
            }
        }

        curCol += matrix[i].cols();
    }

    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    imwrite(path, image, compression_params);

    //imshow("test", image);
    //waitKey();

    return 0;
}

} /* namespace DeepLearning */
