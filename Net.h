#ifndef NET_H_
#define NET_H_

#include "Layer.h"

enum class LearningRateDecayType{
    ExponentialDecay = 0,
    NaturalExpDecay,
    InverseTimeDecay
};

#define D_LEARNING_RATE (0.01)
#define D_LEARNING_RATE_DECAY_TYPE (LearningRateDecayType::ExponentialDecay)
#define D_LEARNING_RATE_DECAY_RATE (0.5)
#define D_LEARNING_RATE_DECAY_STEP (2)
#define D_WEIGHT_DECAY (0.001)
#define D_MOMENTUM (0.6)

namespace DeepLearning {

class Net {
public:
    Net();
    virtual ~Net();

    int AddLayer(Layer *layer);

    int SetLearningRate(float learningRate, LearningRateDecayType decayType, float decayRate, unsigned int decayStep);

    int SetWeightDecayRate(float weightDecay, float momentum);

    int Train(unsigned int epoches, unsigned int batchSize, const vector<vector<MatrixXf>> input, const vector<MatrixXf> lable);

    int Predict(const vector<vector<MatrixXf>> input, vector<MatrixXf> &lable);

    int SaveModel(string path);

private:
    int UpdateLearningRate(unsigned int step, float &learningRate);
    int GenShuffleIndex(vector<unsigned int> &index);

private:
    LearningRateDecayType m_learning_rate_decay_type;
    float m_learning_rate;
    float m_learning_rate_decay_rate;
    float m_learning_rate_decay_step;

    float m_weight_decay;
    float m_momentum;

    vector<Layer *> m_layers;
};

} /* namespace DeepLearning */

#endif /* NET_H_ */
