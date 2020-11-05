#include "Net.h"
#include <random>
namespace DeepLearning {

Net::Net()
    : m_learning_rate_decay_type(D_LEARNING_RATE_DECAY_TYPE),
      m_learning_rate(D_LEARNING_RATE),
      m_learning_rate_decay_rate(D_LEARNING_RATE_DECAY_RATE),
      m_learning_rate_decay_step(D_LEARNING_RATE_DECAY_STEP),
      m_weight_decay(D_WEIGHT_DECAY),
      m_momentum(D_MOMENTUM){
    // TODO Auto-generated constructor stub

    m_layers.clear();
}

Net::~Net() {
    // TODO Auto-generated destructor stub

    m_layers.clear();
}

int Net::AddLayer(Layer *layer){

    this->m_layers.push_back(layer);

    return 0;
}

int Net::SetLearningRate(float learningRate, LearningRateDecayType decayType, float decayRate, unsigned int decayStep){

    this->m_learning_rate = learningRate;
    this->m_learning_rate_decay_type = decayType;
    this->m_learning_rate_decay_rate = decayRate;
    this->m_learning_rate_decay_step = decayStep;

    return 0;
}

int Net::SetWeightDecayRate(float weightDecay, float momentum){

    this->m_weight_decay = weightDecay;
    this->m_momentum = momentum;

    return 0;
}

int Net::Train(unsigned int epoches, unsigned int batchSize, const vector<vector<MatrixXf>> input, const vector<MatrixXf> lable){

    for(unsigned int e = 0; e < epoches; ++e){

        DBG_MSG(("epoch index : %u", e));

        // updata learning rate
        float learnRate = 0;
        UpdateLearningRate(e, learnRate);

        // generate shuffled index
        vector<unsigned int> index(input.size(), 0);
        GenShuffleIndex(index);

        for(unsigned int i = 0; i < input.size(); ++i){
            // forward
            vector<MatrixXf> inputMatrix = input[index[i]];
            for(unsigned int j = 0; j < this->m_layers.size(); ++j){
                vector<MatrixXf> outputMatrix;
                outputMatrix.clear();
                this->m_layers[j]->Forward(inputMatrix, outputMatrix);
                inputMatrix = outputMatrix;
            }

            // backward
            vector<MatrixXf> lableMatrix;
            lableMatrix.push_back(lable[index[i]]);
            for(unsigned int j = 0; j < this->m_layers.size(); ++j){
                vector<MatrixXf> outputMatrix;
                outputMatrix.clear();
                this->m_layers[this->m_layers.size() - j - 1]->Backward(lableMatrix, outputMatrix);
                lableMatrix = outputMatrix;
            }

            // update gradient
            if(0 == (i + 1) % batchSize){
                for(unsigned int j = 0; j < this->m_layers.size(); ++j){
                    this->m_layers[j]->UpdateGradient(batchSize, learnRate, this->m_weight_decay, this->m_momentum);
                }
            }
        }

        // update gradient
        if(input.size() % batchSize){
            for(unsigned int j = 0; j < this->m_layers.size(); ++j){
                this->m_layers[j]->UpdateGradient(input.size() % batchSize, learnRate, this->m_weight_decay, this->m_momentum);
            }
        }
    }

    return 0;
}

int Net::Predict(const vector<vector<MatrixXf>> input, vector<MatrixXf> &lable){

    lable.clear();
    for(unsigned int i = 0; i < input.size(); ++i){
        vector<MatrixXf> inputMatrix = input[i];
        for(unsigned int j = 0; j < this->m_layers.size(); ++j){
            vector<MatrixXf> outputMatrix;
            this->m_layers[j]->Forward(inputMatrix, outputMatrix);
            inputMatrix = outputMatrix;
        }
        if(false == inputMatrix.empty()){
            lable.push_back(inputMatrix[0]);
        }
    }

    return 0;
}

int Net::SaveModel(string path){

    FILE *hFile = fopen(path.c_str(), "w");
    if(NULL != hFile){
        fprintf(hFile, "LayerCount : %lu\n", this->m_layers.size());
        for(unsigned int i = 0; i < this->m_layers.size(); ++i){
            this->m_layers[i]->SaveLayerParameter(hFile);
        }
        fclose(hFile);
    }

    return 0;
}

int Net::UpdateLearningRate(unsigned int step, float &learningRate){

    switch(this->m_learning_rate_decay_type)
    {
        case LearningRateDecayType::ExponentialDecay :
        {
            learningRate = m_learning_rate * powf(m_learning_rate_decay_rate, step / m_learning_rate_decay_step);
            break;
        }
        case LearningRateDecayType::NaturalExpDecay :
        {
            learningRate = m_learning_rate * expf(1 + m_learning_rate_decay_rate * step / m_learning_rate_decay_step);
            break;
        }
        case LearningRateDecayType::InverseTimeDecay :
        {
            learningRate = m_learning_rate / (1 + m_learning_rate_decay_rate * step / m_learning_rate_decay_step);
            break;
        }
        default:
        {
            learningRate = m_learning_rate * powf(m_learning_rate_decay_rate, step / m_learning_rate_decay_step);
            break;
        }
    }

    return 0;
}

int Net::GenShuffleIndex(vector<unsigned int> &index){

    for(unsigned int i = 0; i < index.size(); ++i){
        index[i] = i;
    }

    //static default_random_engine randomEngine(time(NULL));
    static default_random_engine randomEngine(100000);
    //static default_random_engine randomEngine(1000);
    shuffle(index.begin(), index.end(), randomEngine);

    return 0;
}

} /* namespace DeepLearning */
