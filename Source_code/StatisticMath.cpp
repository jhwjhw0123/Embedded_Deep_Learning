#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "StatisticMath.h"

#define PI 3.1415926

using namespace std;
using namespace Eigen;


MatrixXmat sigmoid_func(MatrixXmat x){
    int m_amount = x.rows();
    int n_Dim = x.cols();
    MatrixXmat y(m_amount,n_Dim);
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<n_Dim;j++){
            y(i,j) = (UnitType)(1/(1+exp(-1*(double)(x(i,j)))));
        }
    }

    return y;
}

MatrixXmat tanh_func(MatrixXmat x){
    int m_amount = x.rows();
    int n_Dim = x.cols();
    MatrixXmat y(m_amount,n_Dim);
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<n_Dim;j++){
            y(i,j) = (UnitType)tanh(double(x(i,j)));
        }
    }

    return y;
}
