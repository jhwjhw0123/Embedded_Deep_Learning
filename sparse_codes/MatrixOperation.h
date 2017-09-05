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
MatrixSparseMat mat_max(MatrixSparseMat mat, string flag);
//function that return the minimum element of each row/column
MatrixSparseMat mat_min(MatrixSparseMat mat, string flag);
//function to return the argmax index for the function
MatrixXi mat_argmax(MatrixSparseMat mat, string flag);
//function to return the argmin index for the function
MatrixXi mat_argmin(MatrixSparseMat mat, string flag);
//function to concatenate different matrix
MatrixSparseMat mat_concate(MatrixSparseMat mat_l, MatrixSparseMat mat_r, int axis);
//function to reshape the matrix
MatrixSparseMat mat_reshape(MatrixSparseMat mat, int newRow,int newColumn, int axis);
//function to extract a matrix to a sub-matrix
MatrixSparseMat mat_extract(MatrixSparseMat mat, int start_row, int end_row, int start_column, int end_column);
//function to diagnalize a vector to a matrix
MatrixSparseMat mat_diagnolize(MatrixSparseMat mat);
#endif // MATRIXOPERATION_H_INCLUDED
