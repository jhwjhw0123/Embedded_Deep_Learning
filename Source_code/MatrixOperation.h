#ifndef MATRIXOPERATION_H_INCLUDED
#define MATRIXOPERATION_H_INCLUDED
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include <cmath>
using namespace std;
using namespace Eigen;
/**********Announcement of fuctions********/
//function to find the maximum element row/column wise
MatrixXmat mat_max(MatrixXmat mat, string flag);
//function that return the minimum element of each row/column
MatrixXmat mat_min(MatrixXmat mat, string flag);
//function to return the argmax index for the function
MatrixXi mat_argmax(MatrixXmat mat, string flag);
//function to return the argmin index for the function
MatrixXi mat_argmin(MatrixXmat mat, string flag);
//function to concatenate different matrix
MatrixXmat mat_concate(MatrixXmat mat_l, MatrixXmat mat_r, int axis);
//function to reshape the matrix
MatrixXmat mat_reshape(MatrixXmat mat, int newRow,int newColumn, int axis);
//function to extract a matrix to a sub-matrix
MatrixXmat mat_extract(MatrixXmat mat, int start_row, int end_row, int start_column, int end_column);
//function to diagnalize a vector to a matrix
MatrixXmat mat_diagnolize(MatrixXmat mat);
#endif // MATRIXOPERATION_H_INCLUDED
