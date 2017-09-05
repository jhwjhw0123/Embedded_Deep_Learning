#include <cmath>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "DNN.h"
#include "MatrixOperation.h"
#include "StatisticMath.h"

using namespace std;
using namespace DNN;
using namespace Eigen;

/***********Codes for transformation layer****************/
tranform_layer::tranform_layer(int inDim, int outDim){
    nDim = inDim;
    nNode = outDim;
    w_initial();
    }

tranform_layer::~tranform_layer(){

    }

void tranform_layer::w_initial(){
    MatrixXmat temp_copy_mat(this->nDim, this->nNode);
    for(unsigned int i=0;i<this->nDim;i++){
        for(unsigned int j=0;j<this->nNode;j++){
            //Randomly generate initial parameter w with normal distribution
            temp_copy_mat(i,j) = 0.01*normal_dist_gen_1d<UnitType>(0.0,1.0);
            }
        }
    //Add a row of ones at the bottom as bias
    MatrixXmat initial_bias =  MatrixXmat::Ones(1,this->nNode);
    //concatenate the two matrix
    this->w = mat_concate(temp_copy_mat,initial_bias,0);

    }

/**************Codes for Activation Layer***************/
activation_layer::activation_layer(int nodeDim){
    nNode = nodeDim;
    }

activation_layer::~activation_layer(){

    }

/*************Codes for Loss Layer*********************/
//They are now in the header file


/**************Codes for Linear layer*****************/
Linear_Layer::~Linear_Layer(){

    }

MatrixXmat Linear_Layer::forward_func(MatrixXmat x){
    //x_star is the matrix of x with bias
    int nData = x.rows();
    MatrixXmat data_bias =  MatrixXmat::Ones(nData,1);
    MatrixXmat x_star = mat_concate(x,data_bias,1); //concatenate among the row
    MatrixXmat y = x_star*this->w;
    //assign the input to the class object variable for backpropagation
    this->x = x;

    return y;
    }

MatrixXmat Linear_Layer::backward_func(MatrixXmat dLdy){
    //dLdy is the backpropagation error from the following layer, in shape [m_amount * nNode]
    int nDim = this->w.rows();
    int nNode = this->w.cols();
    int m_amount = dLdy.rows();
    //remove the bias of the weight
    MatrixXmat dydx = mat_extract(this->w,1,nDim-1,1,nNode).transpose();
    MatrixXmat dLdx = dLdy*dydx;
    MatrixXmat dLdw = this->x.transpose()*dLdy;
    MatrixXmat dLdb = MatrixXmat::Ones(1,m_amount)*dLdy;
    this -> gradient = mat_concate(dLdw,dLdb,0);

    return dLdx;
    }


/**************Codes for RELU layer***************/
RELU_Layer::~RELU_Layer(){

    }

MatrixXmat RELU_Layer::forward_func(MatrixXmat x){
    int m_amount = x.rows();
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<this->nNode;j++){
            if(x(i,j)<0){
                x(i,j) = 0;
                }
            }
        }
    this->x = x;

    return x;
    }

MatrixXmat RELU_Layer::backward_func(MatrixXmat dLdy){
    int m_amount = x.rows();
    MatrixXmat dydx = this->x;
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<this->nNode;j++){
            //RELU backprop: the data which is more than zero will be passed and directly pass back
            if(dydx(i,j)<=0){
                dydx(i,j) = 0;
                }
            else{
                dydx(i,j) = 1;
                }
            }
        }
    MatrixXmat dLdx = dLdy.cwiseProduct(dydx);

    return dLdx;
    }


/************Codes for sigmoid layer**************/
sigmoid_layer::~sigmoid_layer(){

    }

MatrixXmat sigmoid_layer::forward_func(MatrixXmat x){
    int m_amount = x.rows();
    MatrixXmat y(m_amount,this->nNode);
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<this->nNode;j++){
            y(i,j) = (UnitType)(1/(1+exp((double)(-x(i,j)))));
            }
        }
    this->x = x;

    return y;
    }

MatrixXmat sigmoid_layer::backward_func(MatrixXmat dLdy){
    int m_amount = x.rows();
    MatrixXmat dLdx(m_amount,this->nNode);
    for(unsigned int i=0;i<m_amount;i++){
        for(unsigned int j=0;j<this->nNode;j++){
            double temp_value = (1/(1+exp((double)(-(this->x(i,j))))))*(1 - (1/(1+exp((double)(-(this->x(i,j)))))))*(double)(dLdy(i,j));
            dLdx(i,j) = (UnitType)(temp_value);
            }
        }

    return dLdx;
    }

/*****************codes for dropout layer*****************/
dropout_layer::~dropout_layer(){
}

//forward function
MatrixXmat dropout_layer::forward_func(MatrixXmat x){
    int nData = x.rows();
    int nDim = x.cols();
    vector<int> ind_list;
    MatrixXmat rst_mat = x;
    while(ind_list.size()<this->dropDim){
        bool same_flag = false;
        int this_ind = rand()%dropDim;
        for(unsigned i=0;i<ind_list.size();i++){
            if (ind_list[i]==this_ind){
                same_flag = true;
                break;
            }
        }
        if (same_flag==false){
            ind_list.push_back(this_ind);
        }
    }
    //assign the dropout id
    this->drop_ind = ind_list;
    for(unsigned int k=0;k<dropDim;k++){
        int drop_dim = ind_list[k];
        for(unsigned int i=0;i<nData;i++){
            rst_mat(i,drop_dim) = 0;
            }
        }

    return rst_mat;
    }

//backward function
MatrixXmat dropout_layer::backward_func(MatrixXmat dLdy){
    int nData = dLdy.rows();
    int dropDim = this->drop_ind.size();
    MatrixXmat dLdx = dLdy;
    for(unsigned int k=0;k<dropDim;k++){
        int drop_dim = this->drop_ind[k];
        for(unsigned int i=0;i<nData;i++){
            dLdx(i,drop_dim) = 0;
            }
        }

    return dLdx;
}

/*************codes for softmax_cross_entropy_layer*************/
//They are wrote in the header file now...

