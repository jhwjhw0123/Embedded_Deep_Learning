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
    MatrixXmat w;
    MatrixXmat x;
    MatrixXmat gradient;
public:
    tranform_layer(int inDim, int outDim);
    ~tranform_layer();
    MatrixXmat Para_copy_store(){
        return w;
    };
    void Para_copy_read(MatrixXmat w_read){
        this->w = w_read;
    };
    void w_initial();
    };

//The base class for all kinds of activation layers
class activation_layer{
protected:
    int nNode;
    MatrixXmat x;
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
    MatrixXmat output;
    MatrixXmat target;
    //Methods
    loss_layer(int outDim){
        this->yDim = outDim;
    };
    ~loss_layer(){
    };
    MatrixXmat process_function(MatrixXmat input_mat);
    decimal norm_l1_loss(MatrixXmat prediction, MatrixXmat logits){
        MatrixXmat dif_mat = prediction-logits;
        return dif_mat.lpNorm<1>();
    };
    decimal norm_l2_loss(MatrixXmat prediction, MatrixXmat logits){
        MatrixXmat dif_mat = prediction-logits;
        return dif_mat.squaredNorm();
    };
    decimal cross_entropy_loss(MatrixXmat prediction, MatrixXmat logits){
        //predictions and logits are all in [m_amount * n_class] shape
        //for classificaion problem, yDim is essentially the n_class
        //consider to add one_hot encoding later
        int m_amount = prediction.rows();
        int n_class = this->yDim;
        decimal cross_entropy_loss = 0;
        decimal single_data_loss = 0;
        for(unsigned int i=0;i<m_amount;i++){
            for(unsigned int j=0;j<n_class;j++){
                single_data_loss -= robust_log<UnitType>(prediction(i,j))*logits(i,j);
                }
            cross_entropy_loss += single_data_loss;
            single_data_loss = 0;
            }
        cross_entropy_loss = cross_entropy_loss/(decimal)m_amount;

        return cross_entropy_loss;
    };
    decimal acc_classfication(MatrixXmat prediction, MatrixXmat logits){
        //prediction and logits in [nData * nDim] shape
        unsigned int nData = prediction.rows();
        unsigned int nDim = prediction.cols();
        unsigned int predict_label = nDim+1;  //make the label impossible to be guessed to right
        unsigned int true_label = nDim+2;
        unsigned int correct_predictions = 0;
        for(unsigned int i=0;i<nData;i++){
            bool logit_flag = false;
            bool predict_flag = false;
            for(unsigned int j=0;j<nDim;j++){
                if(prediction(i,j)==1){
                    predict_label = j;
                    predict_flag = true;
                }
                if(logits(i,j)==1){
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
    MatrixXmat forward_func(MatrixXmat x);
    MatrixXmat backward_func(MatrixXmat dLdy);
    //Update function defines here
    template <typename Opt>
    void train_func(Opt &optimizer,string regulization, UnitType lambda){
        //Currently support sgd
        int nheight = this -> gradient.rows();
        int nwidth = this -> gradient.cols();
        MatrixXmat reg_mat = MatrixXmat::Zero(nheight,nwidth);
        MatrixXi max_ind;
        if(regulization=="None"){
        }
        else if(regulization=="l_1"){
            UnitType threshold = 0.01;
            for(unsigned int i=0;i<nheight;i++){
                for(unsigned int j=0;j<nwidth;j++){
                    if(abs(this->w(i,j))<threshold){
                        reg_mat(i,j) = 0;
                    }
                    else if(this->w(i,j)>0){
                        reg_mat(i,j) = lambda*1;
                    }
                    else{
                        reg_mat(i,j) = lambda*(-1);
                    }
                }
            }
        }
        else if(regulization=="l_2"){
            reg_mat = lambda*this->w;
        }
        else if(regulization=="l_inf"){
            max_ind = mat_argmax(this->w,"row");
            for(unsigned int i=0;i<nheight;i++){
                int this_ind = max_ind(i,0);
                reg_mat(i,this_ind) = lambda*w(i,this_ind);
            }
        }
        MatrixXmat learn_mat = this->gradient + reg_mat;
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
    MatrixXmat forward_func(MatrixXmat x);
    MatrixXmat backward_func(MatrixXmat dLdy);
};

//sigmoid activation layer
class sigmoid_layer:public activation_layer{
public:
    sigmoid_layer(int nodeDim):activation_layer(nodeDim){};
    ~sigmoid_layer();
    MatrixXmat forward_func(MatrixXmat x);
    MatrixXmat backward_func(MatrixXmat dLdy);
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
    MatrixXmat forward_func(MatrixXmat x);
    MatrixXmat backward_func(MatrixXmat dLdy);
};

//softmax cross-entropy loss layer
template <typename decimal>
class soft_max_cross_entropy_layer: public loss_layer<decimal>{
public:
    soft_max_cross_entropy_layer(int outDim):loss_layer<decimal>(outDim){};
    ~soft_max_cross_entropy_layer(){
    };
    MatrixXi output_func(MatrixXmat input, MatrixXmat target){
        int m_amount = input.rows();
        int n_class = this->yDim;
        //soft_max output
        MatrixXmat soft_max_output = this->softmax_func(input);  //This should be [m_amount * nDim]
        decimal loss = this->cross_entropy_loss(soft_max_output,target);
        //get the argmax of the output
        MatrixXi output_labels = mat_argmax(soft_max_output,"row");
        this->output = soft_max_output;
        this->target = target;
        this->loss = loss;

        return output_labels;
    }
    MatrixXmat backward_func(){
        MatrixXmat dLdx = this->output - this->target;

        return dLdx;
    }
    MatrixXmat softmax_func(MatrixXmat x){
        int m_amount = x.rows();
        int n_class = this->yDim;
        decimal norm_exp_value = 0;
        MatrixXmat soft_max_output(m_amount,n_class);
        for(unsigned int i=0;i<m_amount;i++){
            for(unsigned int j=0;j<n_class;j++){
                soft_max_output(i,j) = (UnitType)(exp(double(x(i,j))));
                norm_exp_value += soft_max_output(i,j);
                //cout<<x[i][j]<<" ";
                }
            for(unsigned int j=0;j<n_class;j++){
                soft_max_output(i,j) = soft_max_output(i,j)/norm_exp_value;
                }
            norm_exp_value = 0;
            //cout<<endl;
            }

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
    MatrixXi output_func(MatrixXmat input, MatrixXmat target){
        int m_amount = input.rows();
        int n_class = this->yDim;
        if(input.cols()!=1){
            throw std::invalid_argument("Sigmoid Loss layer could only accept 1-dim input!");
        }
        //sigmoid output
        MatrixXmat sigmoid_output = this->sigmoid_func(input);  //This should be [m_amount * nDim]
        decimal loss = this->cross_entropy_loss(sigmoid_output,target);
        //get the argmax of the output
        MatrixXi output_labels = mat_argmax(sigmoid_output,"row");
        this->output = sigmoid_output;
        this->target = target;
        this->loss = loss;

        return output_labels;
    }
    MatrixXmat backward_func(){
        MatrixXmat dLdx = this->output - this->target;

        return dLdx;
    }
    MatrixXmat sigmoid_func(MatrixXmat x){
        int m_amount = x.rows();
        int n_class = this->yDim;
        decimal norm_exp_value = 0;
        MatrixXmat sigmoid_output(m_amount,n_class);
        for(unsigned int i=0;i<m_amount;i++){
            sigmoid_output(i,0) = (UnitType)(1/(1 + exp(-(double)(x(i,0)))));
            sigmoid_output(i,1) = 1- (sigmoid_output(i,0));
            }

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
    MatrixXmat output_func(MatrixXmat input, MatrixXmat target){
        int m_amount = input.rows();
        int output_dim = this->yDim;
        if(input.cols()!=1){
            throw std::invalid_argument("Regression Loss layer could only accept 1-dim input!");
        }
        decimal loss = this->norm_l2_loss(input,target);
        //get the argmax of the output
        MatrixXmat output_values = input;
        this->output = input;
        this->target = target;
        this->loss = loss;

        return output_values;
    }
    MatrixXmat backward_func(){
        MatrixXmat dLdx = this->output - this->target;

        return dLdx;
    }
    };

};



#endif // DNN_H_INCLUDED
