#include "Minst.h"

namespace DeepLearning {

Minst::Minst() {
    // TODO Auto-generated constructor stub

}

Minst::~Minst() {
    // TODO Auto-generated destructor stub
}

int Minst::ReadImage(const string path, vector<vector<MatrixXf>> &imgs){

    FILE  *hFile = NULL;
    hFile = fopen(path.c_str(),"rb");
    if(hFile == NULL){
        DBG_ERROR(("there is no path : %s\n", path.c_str()));
        return -1;
    }

    imgs.clear();

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    size_t ret = 0;

    ret = fread((char*)&magic_number,sizeof(magic_number), 1, hFile);
    if(0 == ret){
        return -1;
    }
    magic_number = ReverseInt(magic_number);

    ret = fread((char*)&number_of_images,sizeof(number_of_images), 1, hFile);
    if(0 == ret){
        return -1;
    }
    number_of_images = ReverseInt(number_of_images);

    ret = fread((char*)&n_rows,sizeof(n_rows), 1, hFile);
    if(0 == ret){
        return -1;
    }
    n_rows = ReverseInt(n_rows);

    ret = fread((char*)&n_cols,sizeof(n_cols), 1, hFile);
    if(0 == ret){
        return -1;
    }
    n_cols = ReverseInt(n_cols);

    for(int i = 0; i < number_of_images; ++i)
    {
        MatrixXf img = MatrixXf::Zero(n_rows, n_cols);
        for(int j = 0; j < n_rows; ++j)
        {
            for(int k = 0; k < n_cols; ++k)
            {
                unsigned char temp = 0;
                ret = fread((char*) &temp, sizeof(temp), 1, hFile);
                if(0 == ret){
                    return -1;
                }
                img(j, k) = (float)temp/255.0;
            }
        }

        vector<MatrixXf> imgVector;
        imgVector.push_back(img);

        imgs.push_back(imgVector);
    }

    fclose(hFile);

    return 0;
}

int Minst::ReadLable(const string path, vector<MatrixXf> &lables){

    FILE  *hFile = NULL;
    hFile = fopen(path.c_str(),"rb");
    if(hFile == NULL){
        DBG_ERROR(("there is no path : %s\n", path.c_str()));
        return -1;
    }

    lables.clear();

    int magic_number = 0;
    int number_of_labels = 0;
    int label_long = 10;
    size_t ret = 0;

    ret = fread((char*)&magic_number,sizeof(magic_number), 1, hFile);
    if(0 == ret){
        return -1;
    }
    magic_number = ReverseInt(magic_number);

    ret = fread((char*)&number_of_labels,sizeof(number_of_labels), 1, hFile);
    if(0 == ret){
        return -1;
    }

    number_of_labels = ReverseInt(number_of_labels);

    for(int i = 0; i < number_of_labels; ++i)
    {
        MatrixXf lable = MatrixXf::Zero(label_long, 1);
        unsigned char temp = 0;
        ret = fread((unsigned char*)&temp, sizeof(temp), 1, hFile);
        if(0 == ret){
           return -1;
        }
        lable((unsigned int)temp, 0) = 1.0f;
        lables.push_back(lable);
    }

    fclose(hFile);

    return 0;
}

int Minst::ReverseInt(int num)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = num & 255;
    ch2 = (num >> 8) & 255;
    ch3 = (num >> 16) & 255;
    ch4 = (num >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

} /* namespace DeepLearning */
