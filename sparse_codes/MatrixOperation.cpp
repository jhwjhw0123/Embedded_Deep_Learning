/*******************************
*@function: Essential matrix oprerations for the neural network
*@element: Vectors as containers
*@author: Chen Wang, Dept. of Computer Science, University College London
********************************/

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
MatrixSparseMat mat_max(MatrixSparseMat mat, string flag){
//One principle here: I will keep each matrix-like elements as 2-d, even it is a vector (n x 1)
    int nRow = mat.rows();
    int nColumn = mat.cols();
    MatrixXmat mat_dense = MatrixXmat(mat);
    MatrixXmat rst_mat_dense;
    if(flag=="row"){
        rst_mat_dense = mat_dense.rowwise().maxCoeff();
        }
    else if(flag=="column"){
        rst_mat_dense = mat_dense.colwise().maxCoeff();
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        }

    MatrixSparseMat rst_mat = rst_mat_dense.sparseView();
    return rst_mat;
    }

//function that return the minimum element of each row/column
MatrixSparseMat mat_min(MatrixSparseMat mat, string flag){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    MatrixXmat mat_dense = MatrixXmat(mat);
    MatrixXmat rst_mat_dense;
    if(flag=="row"){
        rst_mat_dense = mat_dense.rowwise().minCoeff();
        MatrixSparseMat rst_mat = rst_mat_dense.sparseView();

        return rst_mat;
        }
    else if(flag=="column"){
        rst_mat_dense = mat_dense.colwise().minCoeff();
        MatrixSparseMat rst_mat = rst_mat_dense.sparseView();

        return rst_mat;
        }
    else{
        throw std::invalid_argument("Input mode unrecognized!");
        MatrixSparseMat rst_mat;
        return rst_mat;
        }
    }

//function to return the argmax index for the function
MatrixXi mat_argmax(MatrixSparseMat mat, string flag){
    unsigned int nRow = mat.rows();
    unsigned int nColumn = mat.cols();
    MatrixXmat mat_dense = MatrixXmat(mat);
    if(flag=="row"){
        //find the maximum element of each row
        MatrixSparseMat max_mat(nRow,1);
        MatrixXi rst_mat(nRow,1);
        for(unsigned int i=0;i<nRow;i++){
            MatrixXmat::Index   maxIndex_row;
            max_mat.coeffRef(i,0) = mat_dense.row(i).maxCoeff(&maxIndex_row);
            rst_mat.coeffRef(i,0) = maxIndex_row;
            }

        return rst_mat;
        }
    else if(flag=="column"){
        //find the maximum element of each column
        MatrixSparseMat max_mat(nColumn,1);
        MatrixXi rst_mat(nColumn,1);
        for(unsigned int i=0;i<nColumn;i++){
            MatrixXmat::Index   maxIndex_column;
            max_mat.coeffRef(i,0) = mat_dense.col(i).maxCoeff(&maxIndex_column);
            rst_mat.coeffRef(i,0) = maxIndex_column;
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
MatrixXi mat_argmin(MatrixSparseMat mat, string flag){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    MatrixXmat mat_dense = MatrixXmat(mat);
    if(flag=="row"){
        //find the maximum element of each row
        MatrixSparseMat min_mat(nRow,1);
        MatrixXi rst_mat(nRow,1);
        for(unsigned int i=0;i<nRow;i++){
            MatrixXmat::Index  minIndex_row;
            min_mat.coeffRef(i,0) = mat_dense.row(i).minCoeff(&minIndex_row);
            rst_mat.coeffRef(i,0) = minIndex_row;
            }

        return rst_mat;
        }
    else if(flag=="column"){
        //find the maximum element of each column
        MatrixSparseMat min_mat(nColumn,1);
        MatrixXi rst_mat(nColumn,1);
        for(unsigned int i=0;i<nColumn;i++){
            MatrixXmat::Index   minIndex_column;
            rst_mat.coeffRef(i,0) = mat_dense.col(i).minCoeff(&minIndex_column);
            rst_mat.coeffRef(i,0) = minIndex_column;
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
MatrixSparseMat mat_concate(MatrixSparseMat mat_l, MatrixSparseMat mat_r, int axis){
    int left_row = mat_l.rows();
    int left_column = mat_l.cols();
    int right_row = mat_r.rows();
    int right_column = mat_r.cols();
    MatrixXmat mat_l_dense = MatrixXmat(mat_l);
    MatrixXmat mat_r_dense = MatrixXmat(mat_r);
    if(axis==0){
        //row-wise concatenate
        if(left_column!=right_column){
            throw std::invalid_argument("Cannot concatenate two matrix: dimensionality not matched!");
            return mat_l;
            }
        else{
            MatrixXmat rst_mat_dense(left_row+right_row,left_column);
            rst_mat_dense<<mat_l_dense,mat_r_dense;
            MatrixSparseMat rst_mat = rst_mat_dense.sparseView();

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
            MatrixXmat rst_mat_dense(left_row,left_column+right_column);
            rst_mat_dense<<mat_l_dense,mat_r_dense;
            MatrixSparseMat rst_mat = rst_mat_dense.sparseView();

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
MatrixSparseMat mat_reshape(MatrixSparseMat mat, int newRow, int newColumn, int axis){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    int n_old_element = nRow*nColumn;
    int n_new_element = newRow*newColumn;
    MatrixSparseMat new_mat;
    if(n_old_element!=n_new_element){
        throw std::invalid_argument("The new shape must not change the number of elements of the matrix!");
        return mat;
        }
    if(axis==0){
        new_mat = mat;
        }
    else if(axis==1){
        new_mat = mat.transpose();
        }
    else{
        throw std::invalid_argument("Argument axis must be 0 or 1!");
        }
    new_mat.resize(newRow,newColumn);

    return new_mat;
    }

//function to extract a matrix to a sub-matrix
MatrixSparseMat mat_extract(MatrixSparseMat mat, int start_row, int end_row, int start_column, int end_column){
    int nRow = mat.rows();
    int nColumn = mat.cols();
    MatrixXmat mat_dense = MatrixXmat(mat);
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
    MatrixXmat rst_dense = mat_dense.block(start_row_ind,start_column_ind,row_size,column_size);
    MatrixSparseMat rst_mat = rst_dense.sparseView();

    return rst_mat;
    }

MatrixSparseMat mat_diagnolize(MatrixSparseMat mat){
    MatrixSparseMat temp_mat;
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
    MatrixSparseMat rst_mat = MatrixSparseMat(nDim,nDim);
    for (unsigned int i=0;i<nDim;i++){
        rst_mat.coeffRef(i,i) = temp_mat.coeffRef(0,i);
    }

    return rst_mat;
}
