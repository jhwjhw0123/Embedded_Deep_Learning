#include <iostream>
#include <stdlib.h>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "MatrixOperation.h"

using namespace std;
using namespace Eigen;

//function that return the maximum element of each row/column
MatrixXmat mat_max(MatrixXmat mat, string flag){
//One principle here: I will keep each matrix-like elements as 2-d, even it is a vector (n x 1)
    int nRow = mat.rows();
    int nColumn = mat.cols();
    MatrixXmat rst_mat;
    if(flag=="row"){
        rst_mat = mat.rowwise().maxCoeff();
        }
    else if(flag=="column"){
        rst_mat = mat.colwise().maxCoeff();
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        }
    return rst_mat;
    }

//function that return the minimum element of each row/column
MatrixXmat mat_min(MatrixXmat mat, string flag){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    if(flag=="row"){
        MatrixXmat rst_mat = mat.rowwise().minCoeff();

        return rst_mat;
        }
    else if(flag=="column"){
        MatrixXmat rst_mat = mat.colwise().minCoeff();

        return rst_mat;
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        MatrixXmat rst_mat;
        return rst_mat;
        }
    }

//function to return the argmax index for the function
MatrixXi mat_argmax(MatrixXmat mat, string flag){
    unsigned int nRow = mat.rows();
    unsigned int nColumn = mat.cols();
    if(flag=="row"){
        //find the maximum element of each row
        MatrixXmat max_mat(nRow,1);
        MatrixXi rst_mat(nRow,1);
        for(unsigned int i=0;i<nRow;i++){
            MatrixXmat::Index   maxIndex_row;
            max_mat(i,0) = mat.row(i).maxCoeff(&maxIndex_row);
            rst_mat(i,0) = maxIndex_row;
            }

        return rst_mat;
        }
    else if(flag=="column"){
        //find the maximum element of each column
        MatrixXmat max_mat(nColumn,1);
        MatrixXi rst_mat(nColumn,1);
        for(unsigned int i=0;i<nColumn;i++){
            MatrixXmat::Index   maxIndex_column;
            max_mat(i,0) = mat.col(i).maxCoeff(&maxIndex_column);
            rst_mat(i,0) = maxIndex_column;
            }
        return rst_mat;
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        MatrixXi rst_mat;
        return rst_mat;
        }
    }

//function to return the argmin index for the function
MatrixXi mat_argmin(MatrixXmat mat, string flag){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    if(flag=="row"){
        //find the maximum element of each row
        MatrixXmat min_mat(nRow,1);
        MatrixXi rst_mat(nRow,1);
        for(unsigned int i=0;i<nRow;i++){
            MatrixXmat::Index  minIndex_row;
            min_mat(i,0) = mat.row(i).minCoeff(&minIndex_row);
            rst_mat(i,0) = minIndex_row;
            }

        return rst_mat;
        }
    else if(flag=="column"){
        //find the maximum element of each column
        MatrixXmat min_mat(nColumn,1);
        MatrixXi rst_mat(nColumn,1);
        for(unsigned int i=0;i<nColumn;i++){
            MatrixXmat::Index   minIndex_column;
            rst_mat(i,0) = mat.col(i).minCoeff(&minIndex_column);
            rst_mat(i,0) = minIndex_column;
            }
        return rst_mat;
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        MatrixXi rst_mat;
        return rst_mat;
        }
    }


//function to concatenate different matrix
MatrixXmat mat_concate(MatrixXmat mat_l, MatrixXmat mat_r, int axis){
    int left_row = mat_l.rows();
    int left_column = mat_l.cols();
    int right_row = mat_r.rows();
    int right_column = mat_r.cols();
    if(axis==0){
        //row-wise concatenate
        if(left_column!=right_column){
            throw std::invalid_argument("Cannot concatenate two matrix: dimensionality not matched!");
            return mat_l;
            }
        else{
            MatrixXmat rst_mat(left_row+right_row,left_column);
            rst_mat<<mat_l,mat_r;
            return rst_mat;
            }
        }
    else if(axis==1){
        //colum-wise concatenate
        if(left_row!=right_row){
            throw std::invalid_argument("Cannot concatenate two matrix: dimensionality not matched!");
            return mat_l;
            }
        else{
            MatrixXmat rst_mat(left_row,left_column+right_column);
            rst_mat<<mat_l,mat_r;
            return rst_mat;
            }
        }
    else{
        //invalid
        throw std::invalid_argument("Argument axis must be 0 or 1!");
        return mat_l;
        }
    }

//function to reshape the matrix
MatrixXmat mat_reshape(MatrixXmat mat, int newRow, int newColumn, int axis){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    int n_old_element = nRow*nColumn;
    int n_new_element = newRow*newColumn;
    if(n_old_element!=n_new_element){
        throw std::invalid_argument("The new shape must not change the number of elements of the matrix!");
        return mat;
        }
    if(axis==0){
       mat.resize(newRow,newColumn);
        }
    else if(axis==1){
        mat.transposeInPlace();
        mat.resize(newRow,newColumn);
        }
    else{
        throw std::invalid_argument("Argument axis must be 0 or 1!");
        }
    mat.transposeInPlace();

    return mat;
    }

//function to extract a matrix to a sub-matrix
MatrixXmat mat_extract(MatrixXmat mat, int start_row, int end_row, int start_column, int end_column){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    if(start_row<1||start_column<1){
        throw std::invalid_argument("Start row or column should be more than 1!");
        return mat;
        }
    if(end_row>nRow){
        throw std::invalid_argument("Row extraction out of range!");
        return mat;
        }
    if(end_column>nColumn){
        throw std::invalid_argument("Column extraction out of range!");
        return mat;
        }
    int start_row_ind = start_row-1;
    int start_column_ind = start_column - 1;
    int row_size = end_row - start_row + 1;
    int column_size = end_column - start_column + 1;
    MatrixXmat rst_mat = mat.block(start_row_ind,start_column_ind,row_size,column_size);

    return rst_mat;
    }

MatrixXmat mat_diagnolize(MatrixXmat mat){
    MatrixXmat temp_mat;
    //transfer this into a row vector
    if(mat.rows()>mat.cols()){
        temp_mat = mat.transpose();
    }
    else{
        temp_mat = mat;
    }
    if (mat.rows()!=1){
        throw std::invalid_argument("The input must be a vector");
    }
    int nDim = mat.cols();
    MatrixXmat rst_mat = MatrixXmat::Zero(nDim,nDim);
    for (unsigned int i=0;i<nDim;i++){
        rst_mat(i,i) = temp_mat(0,i);
    }

    return rst_mat;
}
