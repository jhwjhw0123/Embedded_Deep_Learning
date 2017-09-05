#ifndef STATISTICMATH_H_INCLUDED
#define STATISTICMATH_H_INCLUDED
#include<vector>
#include <time.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"

#define PI 3.1415926

using namespace std;
using namespace Eigen;

template <typename decimal>
decimal robust_log(decimal input){
    if(input>=1e-5){
        return (decimal)(log(double(input)));
    }
    else{
        return -12.00;
    }
}

template <typename decimal>
decimal normal_dist_gen_1d (double mean,double variance){
    //firstly using Box¡§CMuller transformation to generate a stardand normal distribution
    decimal rd_num_1 = (decimal)rand() / (decimal)RAND_MAX;
    decimal rd_num_2 = (decimal)rand() / (decimal)RAND_MAX;
    decimal rd_norm_std = (decimal)(sqrt((double)(-2*robust_log<UnitType>(rd_num_1)))*cos((double)(2*PI*rd_num_2)));
    decimal rst_num_norm = variance*rd_norm_std + mean;

    return rst_num_norm;
}

MatrixXmat sigmoid_func(MatrixXmat x);

MatrixXmat tanh_func(MatrixXmat x);

#endif // STATISTICMATH_H_INCLUDED
