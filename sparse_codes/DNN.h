#ifndef DNN_H_INCLUDED
#define DNN_H_INCLUDED
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "Optimizer.h"
#include "StatisticMath.h"
#include "MatrixOperation.h"
/**********************************
*@author:Chen Wang
*@input: Using Eigen package to represent and compute matrix
************************************/

using namespace std;
using namespace Eigen;

namespace DNN{
/*****************Define three kinds of basic layers**********************/
//The base class for all kinds of transformation layers
class tranform_layer{
protected:
    int nDim;   //input dimension
    int nNode;  //output dimension
    MatrixSparseMat w;
    MatrixSparseMat x;
    MatrixSparseMat gradient;
public:
    tranform_layer(int inDim, int outDim);
    ~tranform_layer();
    MatrixXmat Para_copy_store(){
        return MatrixXmat(this->w);
    };
    void Para_copy_read(MatrixXmat w_read){
        this->w = w_read.sparseView();
    };
    void w_initial();
    };

//The base class for all kinds of activation layers
class activation_layer{
protected:
    int nNode;
    MatrixSparseMat x;
public:
    activation_layer(int nodeDim);
    ~activation_layer();
    };

//The base class for all kinds of loss layers
template <typename decimal>
class loss_layer{
protected:
    int yDim;   //network output dimentsion
public:
    //variables
    double loss;
    MatrixSparseMat output;
    MatrixSparseMat target;
    //Methods
    loss_layer(int outDim){
        this->yDim = outDim;
    };
    ~loss_layer(){
    };
    MatrixSparseMat process_function(MatrixSparseMat input_mat);
    decimal norm_l1_loss(MatrixSparseMat prediction, MatrixSparseMat logits){
        MatrixSparseMat dif_mat = prediction-logits;
        return dif_mat.norm();
    };
    decimal norm_l2_loss(MatrixSparseMat prediction, MatrixSparseMat logits){
        MatrixSparseMat dif_mat = prediction-logits;
        return dif_mat.squaredNorm();
    };
    decimal cross_entropy_loss(MatrixSparseMat prediction, MatrixSparseMat logits){
        //predictions and logits are all in [m_amount * n_class] shape
        //for classificaion problem, yDim is essentially the n_class
        //consider to add one_hot encoding later
        int m_amount = prediction.rows();
        int n_class = this->yDim;
        MatrixXmat prediction_dense = MatrixXmat(prediction);
        MatrixXmat logits_dense = MatrixXmat(logits);
        decimal cross_entropy_loss = 0;
        decimal single_data_loss = 0;
        for(unsigned int i=0;i<m_amount;i++){
            for(unsigned int j=0;j<n_class;j++){
                single_data_loss -= robust_log<UnitType>(prediction_dense.coeffRef(i,j))*logits_dense.coeffRef(i,j);
                }
            cross_entropy_loss += single_data_loss;
            single_data_loss = 0;
            }
        cross_entropy_loss = cross_entropy_loss/(decimal)m_amount;

        return cross_entropy_loss;
    };
    decimal acc_classfication(MatrixSparseMat prediction, MatrixSparseMat logits){
        //prediction and logits in [nData * nDim] shape
        unsigned int nData = prediction.rows();
        unsigned int nDim = prediction.cols();
        unsigned int predict_label = nDim+1;  //make the label impossible to be guessed to right
        unsigned int true_label = nDim+2;
        unsigned int correct_predictions = 0;
        MatrixXmat prediction_dense = MatrixXmat(prediction);
        MatrixXmat logits_dense = MatrixXmat(logits);
        for(unsigned int i=0;i<nData;i++){
            bool logit_flag = false;
            bool predict_flag = false;
            for(unsigned int j=0;j<nDim;j++){
                if(prediction_dense.coeffRef(i,j)==1){
                    predict_label = j;
                    predict_flag = true;
                }
                if(logits_dense.coeffRef(i,j)==1){
                    true_label = j;
                    logit_flag = true;
                }
                if((predict_flag==true)&&(logit_flag == true)){
                    break;
                }
            }
            if(predict_label==true_label){
                correct_predictions += 1;
            }
        }
        decimal accuracy = (decimal)((double)correct_predictions)/(decimal)((double)nData);

        return accuracy;
    };
    };

/*****************Define practical-used layers**********************/

//linear transformation layer
class Linear_Layer:public tranform_layer{
public:
    Linear_Layer(int inDim,int nodeDim):tranform_layer(inDim,nodeDim){};
    ~Linear_Layer();
    //Pass the starting index of the matrix to pass a 2-d array
    MatrixSparseMat forward_func(MatrixSparseMat x);
    MatrixSparseMat backward_func(MatrixSparseMat dLdy);
    //Update function defines here
    template <typename Opt>
    void train_func(Opt &optimizer,string regulization, UnitType lambda){
        //Currently support sgd
        int nheight = this -> gradient.rows();
        int nwidth = this -> gradient.cols();
        MatrixXmat reg_mat_dense = MatrixXmat::Zero(nheight,nwidth);
        MatrixXmat w_dense = MatrixXmat(this->w);
        MatrixXi max_ind;
        if(regulization=="None"){
        }
        else if(regulization=="l_1"){
            UnitType threshold = 0.01;
            for(unsigned int i=0;i<nheight;i++){
                for(unsigned int j=0;j<nwidth;j++){
                    if(abs(w_dense.coeffRef(i,j))<threshold){
                        reg_mat_dense.coeffRef(i,j) = 0;
                    }
                    else if(w_dense.coeffRef(i,j)>0){
                        reg_mat_dense.coeffRef(i,j) = lambda*1;
                    }
                    else{
                        reg_mat_dense.coeffRef(i,j) = lambda*(-1);
                    }
                }
            }
        }
        else if(regulization=="l_2"){
            reg_mat_dense = lambda*w_dense;
        }
        else if(regulization=="l_inf"){
            max_ind = mat_argmax(this->w,"row");
            for(unsigned int i=0;i<nheight;i++){
                int this_ind = max_ind(i,0);
                reg_mat_dense.coeffRef(i,this_ind) = lambda*w_dense.coeffRef(i,this_ind);
            }
        }
        MatrixSparseMat reg_mat = reg_mat_dense.sparseView();
        MatrixSparseMat learn_mat = this->gradient + reg_mat;
        optimizer.get_learn_mat(learn_mat,0);
        this -> w = optimizer.update_function(this->w);
        }
    //void train_func_momentum(double learning_rate,double gamma);
};

//RELU activation layer
class RELU_Layer:public activation_layer{
public:
    RELU_Layer(int nodeDim):activation_layer(nodeDim){};
    ~RELU_Layer();
    MatrixSparseMat forward_func(MatrixSparseMat x);
    MatrixSparseMat backward_func(MatrixSparseMat dLdy);
};

//sigmoid activation layer
class sigmoid_layer:public activation_layer{
public:
    sigmoid_layer(int nodeDim):activation_layer(nodeDim){};
    ~sigmoid_layer();
    MatrixSparseMat forward_func(MatrixSparseMat x);
    MatrixSparseMat backward_func(MatrixSparseMat dLdy);
    };

//dropout layer for regularization
class dropout_layer:public activation_layer{
protected:
    int dropDim;
    vector<int> drop_ind;
public:
    dropout_layer(int nodeDim, int dropDim):activation_layer(nodeDim){
        this->dropDim = dropDim;
        if(dropDim>this->nNode){
            this->dropDim = nNode;
            }
        }
    ~dropout_layer();
    MatrixSparseMat forward_func(MatrixSparseMat x);
    MatrixSparseMat backward_func(MatrixSparseMat dLdy);
};

//softmax cross-entropy loss layer
template <typename decimal>
class soft_max_cross_entropy_layer: public loss_layer<decimal>{
public:
    soft_max_cross_entropy_layer(int outDim):loss_layer<decimal>(outDim){};
    ~soft_max_cross_entropy_layer(){
    };
    MatrixXi output_func(MatrixSparseMat input, MatrixSparseMat target){
        int m_amount = input.rows();
        int n_class = this->yDim;
        //soft_max output
        MatrixSparseMat soft_max_output = this->softmax_func(input);  //This should be [m_amount * nDim]
        decimal loss = this->cross_entropy_loss(soft_max_output,target);
        //get the argmax of the output
        MatrixXi output_labels = mat_argmax(soft_max_output,"row");
        this->output = soft_max_output;
        this->target = target;
        this->loss = loss;

        return output_labels;
    }
    MatrixSparseMat backward_func(){
        MatrixSparseMat dLdx = this->output - this->target;

        return dLdx;
    }
    MatrixSparseMat softmax_func(MatrixSparseMat x){
        int m_amount = x.rows();
        int n_class = this->yDim;
        decimal norm_exp_value = 0;
        MatrixXmat soft_max_output_dense = MatrixXmat::Zero(m_amount,n_class);
        for(unsigned int i=0;i<m_amount;i++){
            for(unsigned int j=0;j<n_class;j++){
                soft_max_output_dense.coeffRef(i,j) = (UnitType)(exp(double(x.coeffRef(i,j))));
                norm_exp_value += soft_max_output_dense.coeffRef(i,j);
                //cout<<x[i][j]<<" ";
                }
            for(unsigned int j=0;j<n_class;j++){
                soft_max_output_dense.coeffRef(i,j) = soft_max_output_dense.coeffRef(i,j)/norm_exp_value;
                }
            norm_exp_value = 0;
            //cout<<endl;
            }

        MatrixSparseMat soft_max_output = soft_max_output_dense.sparseView();

        return soft_max_output;
    }
    };

//sigmoid cross-entropy loss layer (aminly for two-class probability loss)
template <typename decimal>
class sigmoid_cross_entropy_layer: public loss_layer<decimal>{
public:
    sigmoid_cross_entropy_layer():loss_layer<decimal>(){
        this->yDim = 2;
    };
    ~sigmoid_cross_entropy_layer(){
    };
    MatrixXi output_func(MatrixSparseMat input, MatrixSparseMat target){
        int m_amount = input.rows();
        int n_class = this->yDim;
        if(input.cols()!=1){
            throw std::invalid_argument("Sigmoid Loss layer could only accept 1-dim input!");
        }
        //sigmoid output
        MatrixSparseMat sigmoid_output = this->sigmoid_func(input);  //This should be [m_amount * nDim]
        decimal loss = this->cross_entropy_loss(sigmoid_output,target);
        //get the argmax of the output
        MatrixXi output_labels = mat_argmax(sigmoid_output,"row");
        this->output = sigmoid_output;
        this->target = target;
        this->loss = loss;

        return output_labels;
    }
    MatrixSparseMat backward_func(){
        int m_amount = this->output.rows();
        MatrixXmat Ones_mat_dense = MatrixXmat::Ones(m_amount,this->yDim);
        MatrixSparseMat Ones_mat = Ones_mat_dense.sparseView();
        MatrixSparseMat dLdx =  this->target.cwiseProduct((Ones_mat-this->output));

        return dLdx;
    }
    MatrixSparseMat sigmoid_func(MatrixSparseMat x){
        int m_amount = x.rows();
        int n_class = this->yDim;
        decimal norm_exp_value = 0;
        MatrixXmat sigmoid_output_dense = MatrixXmat::Zero(m_amount,n_class);
        for(unsigned int i=0;i<m_amount;i++){
            sigmoid_output_dense.coeffRef(i,0) = (UnitType)(1/(1 + exp(-(double)(x.coeffRef(i,0)))));
            sigmoid_output_dense.coeffRef(i,1) = 1- (sigmoid_output_dense.coeffRef(i,0));
            }
        MatrixSparseMat sigmoid_output = sigmoid_output_dense.sparseView();

        return sigmoid_output;
    }
    };

//Regression loss layer
template <typename decimal>
class l2_norm_loss_layer: public loss_layer<decimal>{
public:
    l2_norm_loss_layer():loss_layer<decimal>(){
        this->yDim = 1;    //regression only accept 1-dim
    };
    ~l2_norm_loss_layer(){
    };
    MatrixSparseMat output_func(MatrixSparseMat input, MatrixSparseMat target){
        int m_amount = input.rows();
        int output_dim = this->yDim;
        if(input.cols()!=1){
            throw std::invalid_argument("Regression Loss layer could only accept 1-dim input!");
        }
        decimal loss = this->norm_l2_loss(input,target);
        //get the argmax of the output
        MatrixSparseMat output_values = input;
        this->output = input;
        this->target = target;
        this->loss = loss;

        return output_values;
    }
    MatrixSparseMat backward_func(){
        MatrixSparseMat dLdx = this->output - this->target;

        return dLdx;
    }
    };

};



#endif // DNN_H_INCLUDED
