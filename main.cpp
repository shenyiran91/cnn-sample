#include <stdio.h>

#include "Net.h"
#include "ConvolutionLayer.h"
#include "FullConnect.h"
#include "PoolingLayer.h"
#include "ReluLayer.h"
#include "SigmoidLayer.h"
#include "TanhLayer.h"
#include "SoftmaxLayer.h"

#include "Minst.h"


#include "DebugPrint.h"

using namespace DeepLearning;

// network structure
#define FULL_NET (1)
#define SAMPLE_NET (2)
#define SOFTMAX_NET (3)
#define NET_TYPE FULL_NET

// data set
#define MINST (1)

#define DATA_SET MINST

int CalAccuracy(vector<MatrixXf> testLables, vector<MatrixXf> outLables)
{
    int count = 0;
    for(unsigned int i = 0; i < testLables.size(); ++i){
        int maxLableIndex = 0, maxOutIndex = 0;

        for(unsigned int j = 0; j < testLables[i].size(); ++j){
            if(testLables[i](j, 0) > testLables[i](maxLableIndex, 0)){
                maxLableIndex = j;
            }
        }

        for(unsigned int j = 0; j < outLables[i].size(); ++j){
            if(outLables[i](j, 0) > outLables[i](maxOutIndex, 0)){
                maxOutIndex = j;
            }
        }

        if(maxLableIndex == maxOutIndex){
            count += 1;
        }
    }

    float rate = (float)count/ testLables.size();
    DBG_MSG(("rate : %f\n", rate));

    return 0;
}


int main()
{
#if (NET_TYPE == SOFTMAX_NET)
    Net cnnNet;
    FullConnect *f1 = new FullConnect(10);
    SoftmaxLayer *s2 = new SoftmaxLayer(10);

    cnnNet.AddLayer(f1);
    cnnNet.AddLayer(s2);
#elif (NET_TYPE == SAMPLE_NET)
    Net cnnNet;
    ConvolutionLayer *c1 = new ConvolutionLayer(5, 6, 1, MatrixXi::Ones(6, 1), PadType::Full);
    ReluLayer *r2 = new ReluLayer();
    PoolingLayer *p3 = new PoolingLayer(2, PoolType::Max);
    FullConnect *f4 = new FullConnect(10);
    SoftmaxLayer *s5 = new SoftmaxLayer(10);

    cnnNet.AddLayer(c1);
    cnnNet.AddLayer(r2);
    cnnNet.AddLayer(p3);
    cnnNet.AddLayer(f4);
    cnnNet.AddLayer(s5);
#else
    MatrixXi linkMatrix(16, 6);
    linkMatrix << 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 0,
        0, 0, 0, 1, 1, 1,
        1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 1,
        1, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1,
        1, 0, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1;

    Net cnnNet;
#if (DATA_SET == MINST)
    ConvolutionLayer *c1 = new ConvolutionLayer(5, 6, 1, MatrixXi::Ones(6, 1), PadType::Full);
#else
    ConvolutionLayer *c1 = new ConvolutionLayer(5, 6, 3, MatrixXi::Ones(6, 3), PadType::Full);
#endif
    ReluLayer *r2 = new ReluLayer();
    //SigmoidLayer *r2 = new SigmoidLayer();
    PoolingLayer *p3 = new PoolingLayer(2, PoolType::Max);
    ConvolutionLayer *c4 = new ConvolutionLayer(5, 16, 1, linkMatrix, PadType::Full);
    ReluLayer *r5 = new ReluLayer();
    //TanhLayer *r5 = new TanhLayer();
    PoolingLayer *p6 = new PoolingLayer(2, PoolType::Max);
    FullConnect *f7 = new FullConnect(10);
    SoftmaxLayer *s8 = new SoftmaxLayer(10);

    cnnNet.AddLayer(c1);
    cnnNet.AddLayer(r2);
    cnnNet.AddLayer(p3);
    cnnNet.AddLayer(c4);
    cnnNet.AddLayer(r5);
    cnnNet.AddLayer(p6);
    cnnNet.AddLayer(f7);
    cnnNet.AddLayer(s8);
#endif

    // load data for train
    vector<vector<MatrixXf>> trainImgs;
    vector<MatrixXf> trainLabels;
#if (DATA_SET == MINST)
    Minst::ReadImage("./mnist/train-images-idx3-ubyte", trainImgs);
    Minst::ReadLable("./mnist/train-labels-idx1-ubyte", trainLabels);
#endif

    cnnNet.SetLearningRate(0.01, LearningRateDecayType::ExponentialDecay, 0.6, 1);
    cnnNet.SetWeightDecayRate(0.001, 0.9);

    cnnNet.Train(5, 50, trainImgs, trainLabels);

    // load data for test
    vector<vector<MatrixXf>> testImgs;
    vector<MatrixXf> testLabels, outLabels;
#if (DATA_SET == MINST)
    Minst::ReadImage("./mnist/t10k-images-idx3-ubyte", testImgs);
    Minst::ReadLable("./mnist/t10k-labels-idx1-ubyte", testLabels);

#endif

    cnnNet.SaveModel("model.txt");

    //cnnNet.Predict(trainInput, outLables);
    //CalAccuracy(trainLables, outLables);

    cnnNet.Predict(testImgs, outLabels);
    CalAccuracy(testLabels, outLabels);

    return 0;
}
