#ifndef MINST_H_
#define MINST_H_

#include <vector>
#include <Eigen/Dense>

#include "DebugPrint.h"

using namespace std;
using namespace Eigen;

namespace DeepLearning {

class Minst {
public:
    Minst();
    virtual ~Minst();

    static int ReadImage(const string path, vector<vector<MatrixXf>> &imgs);
    static int ReadLable(const string path, vector<MatrixXf> &lables);

private:
    static int ReverseInt(int num);
};

} /* namespace DeepLearning */

#endif /* MINST_H_ */
