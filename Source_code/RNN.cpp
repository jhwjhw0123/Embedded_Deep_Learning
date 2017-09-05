/**********************
*@function: Recurrent Neural Network Source Codes
*@author: Chen Wang, Dept. of Computer Science, University College London
*@version: 0.0.1
***********************/
#include <cmath>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdexcept>
#include "RNN.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "MatrixOperation.h"
#include "StatisticMath.h"

using namespace std;
using namespace RNN;
using namespace Eigen;

/**********Codes of functions to reshape the data for RNN************/
MatrixXmat ouput_reshape_rnn(vector<MatrixXmat> data){
    //goal: reshape [nData * nLength * nDim] data into [nData * New_Dim]
    //transform among rows
    int nData = data.size();
    int nLength = data[0].rows();
    int nDim = data[0].cols();
    int new_Dim = nLength*nDim;
    //construct the output vector
    MatrixXmat rst_mat = MatrixXmat::Zero(nData,new_Dim);
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nLength;j++){
            for(unsigned int k=0;k<nDim;k++){
                int this_ind = j*nDim + k;
                rst_mat(i,this_ind) = data[i](j,k);
            }
        }
    }

    return rst_mat;
}
vector<MatrixXmat> input_reshape_rnn(MatrixXmat data, int nLength, int nDim){
    //goal: reshape [nData * nDim] data into [nData * nLength * nDim]
    int nData = data.rows();
    int n_prev_Dim = data.cols();
    if(n_prev_Dim!=nLength*nDim){
        throw std::invalid_argument("The shape of the data must remains unchanged!");
    }
    vector<MatrixXmat> rst_mat(nData,MatrixXmat::Zero(nLength,nDim));
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nLength;j++){
            for(unsigned int k=0;k<nDim;k++){
                int this_ind = j*nDim + k;
                rst_mat[i](j,k) = data(i,this_ind);
            }
        }
    }

    return rst_mat;
}

MatrixXmat ind_data_exact(vector<MatrixXmat> data, int index){
    //input: [nData * nLength * nDim]
    //output: [nDta * nDim], data of the selected time index
    //here index start from 1
    int nData = data.size();
    int nLength = data[0].rows();
    int nDim = data[0].cols();
    if((index>nLength)||(index<1)){
         throw std::invalid_argument("The index of sequence could not exceed the length of the data or less than 1!");
    }
    MatrixXmat selected_data = MatrixXmat::Zero(nData,nDim);
    int input_ind = index-1;
    for(unsigned int i=0;i<nData;i++){
        selected_data.row(i) = data[i].row(input_ind);
    }

    return selected_data;
}

/*******Codes for the base class of RNN******/
Recurrent_net::Recurrent_net(int input_Dim, int rnn_Dim, int output_Dim){
    this->input_Dim = input_Dim;
    this->rnn_Dim = rnn_Dim;
    this->output_Dim = output_Dim;
    this->w_initialize();
}

Recurrent_net::~Recurrent_net(){
}

void Recurrent_net::w_initialize(){
    MatrixXmat temp_w_input(this->input_Dim,this->rnn_Dim);
    MatrixXmat temp_u_trans(this->rnn_Dim,this->rnn_Dim);
    MatrixXmat temp_v_out(this->rnn_Dim,this->output_Dim);
    //Using the following routine to save computational time
    for(unsigned int i=0;i<this->input_Dim;i++){
        for(unsigned int j=0;j<this->rnn_Dim;j++){
            temp_w_input(i,j) = 0.001*normal_dist_gen_1d<UnitType>(0,1);
            if(i==0){
                for(unsigned int k=0;k<this->rnn_Dim;k++){
                    temp_u_trans(j,k) = 0.001*normal_dist_gen_1d<UnitType>(0,1);
                    if(j==0){
                        for(unsigned int p=0;p<this->output_Dim;p++){
                            temp_v_out(k,p) = 0.001*normal_dist_gen_1d<UnitType>(0,1);
                        }
                    }
                }
            }
        }
    }
    this->w_input = temp_w_input;
    this->u_tranform = temp_u_trans;
    this->v_output = temp_v_out;
}

void Recurrent_net::gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind){

}

//compute the gradient of variables
MatrixXmat Recurrent_net::get_para_gradient(MatrixXmat dLdy, string variable, int start_time){
    MatrixXmat rst_mat;

    return rst_mat;
}

//recursively compute the matrix
MatrixXmat Recurrent_net::recursive_func(MatrixXmat mat, int ite_time){
    MatrixXmat rst_mat;

    return rst_mat;
}

MatrixXmat Recurrent_net::dLdtheta_compute(int t, string mode){
    MatrixXmat rst_mat;

    return rst_mat;
}

/****************Codes for basic rnn network*****************/
basic_rnn::~basic_rnn(){
}

//forward routine
vector<MatrixXmat> basic_rnn::forward_func(vector<MatrixXmat> x){
    //input x should be: [nData * nLength * input_dim]
    int nData = x.size();
    this->rnn_length = x[0].rows();
    int nDim = x[0].cols();
    this->x = x;
    //output matrix: [nData * rnn_length * output_Dim]
    vector<MatrixXmat> rst_mat(nData,MatrixXmat::Zero(rnn_length,output_Dim));
    MatrixXmat selected_output; //temporarily store the values of output
    MatrixXmat selected_data;   //the selected data from all the data
    MatrixXmat current_data;
    MatrixXmat current_state;
    MatrixXmat data_state;
    vector<MatrixXmat> iteratoin_state;    //The state [nData * nLength * n_Dim] of the current iteration
    //throw an exception if the input data is not properly shaped
    if(nDim!=this->input_Dim){
        throw std::invalid_argument("The input data must has a dimension same to the initilization dimension!");
        return rst_mat;
    }
    //loop over all data
    for(unsigned int i=0;i<nData;i++){
        selected_data = x[i];
        //cout<<"haha"<<endl;
        for(unsigned int j=0;j<rnn_length;j++){
            current_data = mat_extract(selected_data,j+1,j+1,1,nDim);
            //cout<<"hehe"<<endl;
            if(j==0){
                current_state = current_data*this->w_input;
            }
            else{
                current_state = (current_data*this->w_input)+(this->rnn_state*this->u_tranform);
            }
            //cout<<"zjr"<<endl;
            //store the overall states as [nData * nlength * n_Hiddendim]
            //and sotre the current state
            this->rnn_state = current_state;
            if(selected_output.rows()==0){
                selected_output = this->rnn_state*v_output;
                data_state = this->rnn_state;
            }
            else{
                selected_output = mat_concate(selected_output,(this->rnn_state*v_output),0);
                data_state = mat_concate(data_state,this->rnn_state,0);
            }
        }
        rst_mat[i] = selected_output;
        selected_output.resize(0,0);
        iteratoin_state.push_back(data_state);
        data_state.resize(0,0);
    }
    this->states = iteratoin_state;

    return rst_mat;
}

//backward routine to get the gradient
void basic_rnn::gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind){
    //input_size: [nData * nlength * nDim]
    //get the variables
    int nData = dLdy.size();
    int nLength = dLdy[0].rows();
    int input_dim = this->w_input.rows();
    int hidden_dim = this->u_tranform.rows();
    int output_dim = this->v_output.cols();
    //if the length doesn't math, throw exception
    if(nLength!=y_ind.size()){
        throw std::invalid_argument("The input index must have the same length as the number of backpropagated data!");
    }
    MatrixXmat temp_dLdw = MatrixXmat::Zero(input_dim,hidden_dim);
    MatrixXmat temp_dLdU = MatrixXmat::Zero(hidden_dim,hidden_dim);
    MatrixXmat temp_dLdv = MatrixXmat::Zero(hidden_dim,output_dim);
    MatrixXmat current_dLdy;
    MatrixXmat current_dydv;
    for(unsigned int i=0;i<nLength;i++){
        int t = y_ind[i];
        current_dLdy = ind_data_exact(dLdy,i+1);
        current_dydv = (ind_data_exact(this->states,t)).transpose();
        temp_dLdw = temp_dLdw + get_para_gradient(current_dLdy,"W_input",t);
        temp_dLdU = temp_dLdU + get_para_gradient(current_dLdy,"U_transform",t);
        temp_dLdv = temp_dLdv + current_dydv*current_dLdy;
    }
    this->dLdw = temp_dLdw;
    this->dLdu = temp_dLdU;
    this->dLdv = temp_dLdv;
}

MatrixXmat basic_rnn::get_para_gradient(MatrixXmat dLdy, string variable, int start_time){
    //input: [nData * n_out_dim] dLdy matrix
    //output: dLd(.), the gradient of the current time-index back propagation
    int nLength = this->x[0].rows();
    int input_dim = this->w_input.rows();
    int hidden_dim = this->u_tranform.rows();
    int output_dim = this->v_output.cols();
    //declare the variables of the gradient matrix
    int gradient_mat_width;
    int gradient_mat_length;
    if(variable=="W_input"){
        gradient_mat_width = input_dim;
        gradient_mat_length = hidden_dim;
    }
    else if(variable=="U_transform"){
        gradient_mat_width = hidden_dim;
        gradient_mat_length = hidden_dim;
    }
    else{
        throw std::invalid_argument("The type of the target variable not recognized!");
    }
    MatrixXmat dLd_theta = MatrixXmat::Zero(gradient_mat_width,gradient_mat_length);
    MatrixXmat current_dLd_theta;
    MatrixXmat temp_mat;
    for(unsigned int t=start_time;t>0;t--){
        int ite_time = start_time-t;
        if(variable=="W_input"){
            current_dLd_theta = this->dLdtheta_compute(t,"W_input");//mat_trans(ind_data_exact(this->x,t));
            }
        else if(variable=="U_transform"){
            if(t>1){
                current_dLd_theta = this->dLdtheta_compute(t-1,"U_transform");//mat_trans(ind_data_exact(this->states,t-1));
                }
            }
        if((variable=="W_input")||(t>1)){
            temp_mat = current_dLd_theta*dLdy*(this->v_output.transpose());
            temp_mat = recursive_func(temp_mat,ite_time);
            dLd_theta = dLd_theta+temp_mat;
        }
    }

    return dLd_theta;
}

MatrixXmat basic_rnn::dLdtheta_compute(int t, string mode){
    MatrixXmat rst_mat;
    if(mode=="W_input"){
        rst_mat = ind_data_exact(this->x,t).transpose();
    }
    else if(mode=="U_transform"){
        rst_mat = ind_data_exact(this->states,t).transpose();
    }
    else{
        throw std::invalid_argument("Gradient compute mode unrecognized!");
    }

    return rst_mat;
}

MatrixXmat basic_rnn::recursive_func(MatrixXmat mat, int ite_time){
    int mat_height = mat.rows();
    int mat_width = mat.cols();
    MatrixXmat rst_mat;
    MatrixXmat dHdh_mat = (this->u_tranform).transpose();
    if(ite_time==0){
        rst_mat = mat;
    }
    else if(ite_time==1){
        rst_mat = mat*dHdh_mat;
    }
    else if(ite_time>1){
        rst_mat = recursive_func(mat,ite_time-1)*dHdh_mat;
    }
    else{
        throw std::invalid_argument("The input iteratoin time is invalid!");
    }

    return rst_mat;
}

/***************Codes for LSTM recurrent neural network*************/
lstm_rnn::~lstm_rnn(){

}

void lstm_rnn::w_initialize(){
    Recurrent_net::w_initialize();
    int combined_len = this->input_Dim+this->rnn_Dim;
    MatrixXmat temp_w_f(combined_len,this->rnn_Dim);
    MatrixXmat temp_b_f = MatrixXmat::Ones(1,this->rnn_Dim);
    MatrixXmat temp_w_i(combined_len,this->rnn_Dim);
    MatrixXmat temp_b_i = MatrixXmat::Ones(1,this->rnn_Dim);
    MatrixXmat temp_w_c(combined_len,this->rnn_Dim);
    MatrixXmat temp_b_c = MatrixXmat::Ones(1,this->rnn_Dim);
    MatrixXmat temp_w_o(combined_len,this->rnn_Dim);
    MatrixXmat temp_b_o = MatrixXmat::Ones(1,this->rnn_Dim);
    for(unsigned int i=0;i<combined_len;i++){
        for(unsigned int j=0;j<this->rnn_Dim;j++){
            temp_w_f(i,j) = 0.1*normal_dist_gen_1d<UnitType>(0,1);
            temp_w_i(i,j) = 0.1*normal_dist_gen_1d<UnitType>(0,1);
            temp_w_c(i,j) = 0.1*normal_dist_gen_1d<UnitType>(0,1);
            temp_w_o(i,j) = 0.1*normal_dist_gen_1d<UnitType>(0,1);
        }
    }
    this->w_f = temp_w_f;
    this->b_f = temp_b_f;
    this->w_i = temp_w_i;
    this->b_i = temp_b_i;
    this->w_c = temp_w_c;
    this->b_c = temp_b_c;
    this->w_o = temp_w_o;
    this->b_o = temp_b_o;
}

//forward function
vector<MatrixXmat> lstm_rnn::forward_func(vector<MatrixXmat> x){
     //input x should be: [nData * nLength * input_dim]
    int nData = x.size();
    this->rnn_length = x[0].rows();
    int nDim = x[0].cols();
    this->x = x;
    //output matrix: [nData * rnn_length * output_Dim]
    vector<MatrixXmat> rst_mat(nData,MatrixXmat::Zero(rnn_length,output_Dim));
    MatrixXmat selected_output; //temporarily store the values of output
    //data to input to cell
    MatrixXmat selected_data;   //the selected data from all the data
    MatrixXmat current_data;
    MatrixXmat concate_input;
    //in-cell variables
    MatrixXmat forget_input;    //forget gate
    MatrixXmat i_t_input;       //input gate i_t
    MatrixXmat c_t_input;       //input gate C_t
    MatrixXmat o_t_input;       //output gate
    //variables to store the states
    MatrixXmat C_t_status;
    MatrixXmat current_state;
    //containers to store the information temporarily
    MatrixXmat data_concate;
    MatrixXmat data_state;
    MatrixXmat data_C_t;
    MatrixXmat data_C_in;
    MatrixXmat data_f_t;
    MatrixXmat data_i_t;
    MatrixXmat data_o_t;
    vector<MatrixXmat> iteratoin_concate;  //[nData * nLength * input_dim + hidden_dim]
    vector<MatrixXmat> iteratoin_state;    //The state [nData * nLength * n_Dim] of the current iteration
    vector<MatrixXmat> iteratoin_C_t;      //The hidden state C_t should be in [nData * nLength * n_Dim]
    vector<MatrixXmat> iteratoin_C_in;     //The hidden state C_t should be in [nData * nLength * n_Dim]
    vector<MatrixXmat> iteratoin_f_t;     //The hidden state C_t should be in [nData * nLength * n_Dim]
    vector<MatrixXmat> iteratoin_i_t;     //The hidden state C_t should be in [nData * nLength * n_Dim]
    vector<MatrixXmat> iteratoin_o_t;     //The hidden state C_t should be in [nData * nLength * n_Dim]
    //throw an exception if the input data is not properly shaped
    if(nDim!=this->input_Dim){
        throw std::invalid_argument("The input data must has a dimension same to the initilization dimension!");
        return rst_mat;
    }
    //loop over all data
    for(unsigned int i=0;i<nData;i++){
        selected_data = x[i];  //the selected [rnn_length * input_dim] data
        this->rnn_state = MatrixXmat::Zero(1,this->rnn_Dim);  //initial states
        //looping over time
        for(unsigned int j=0;j<this->rnn_length;j++){
            current_data = mat_extract(selected_data,j+1,j+1,1,nDim);
            //get the concatenated input
            concate_input = mat_concate(this->rnn_state,current_data,1);  //[zeros x_input]
            forget_input = sigmoid_func(concate_input*this->w_f+this->b_f);
            i_t_input = sigmoid_func(concate_input*this->w_i+this->b_i);
            c_t_input = tanh_func(concate_input*this->w_c+this->b_c);
            C_t_status = forget_input.cwiseProduct(this->rnn_state) + i_t_input.cwiseProduct(c_t_input);
            o_t_input = concate_input*this->w_o + this->b_o;
            current_state = (tanh_func(C_t_status)).cwiseProduct(o_t_input);
            this->rnn_state = current_state;
            if(selected_output.rows()==0){
                selected_output = this->rnn_state*v_output;
                data_concate = concate_input;
                data_state = this->rnn_state;
                data_f_t = forget_input;
                data_i_t = i_t_input;
                data_C_in = c_t_input;
                data_C_t = C_t_status;
                data_o_t = o_t_input;
            }
            else{
                selected_output = mat_concate(selected_output,(this->rnn_state*v_output),0);
                data_concate = mat_concate(data_concate,concate_input,0);
                data_state = mat_concate(data_state,this->rnn_state,0);
                data_f_t = mat_concate(data_f_t,forget_input,0);
                data_i_t = mat_concate(data_i_t,i_t_input,0);
                data_C_in = mat_concate(data_C_in,c_t_input,0);
                data_C_t = mat_concate(data_C_t,C_t_status,0);
                data_o_t = mat_concate(data_o_t,o_t_input,0);
            }
        }
        rst_mat[i] = selected_output;
        selected_output.resize(0,0);
        iteratoin_state.push_back(data_state);
        data_state.resize(0,0);
        iteratoin_C_t.push_back(data_C_t);
        data_C_t.resize(0,0);
        iteratoin_C_in.push_back(data_C_in);
        data_C_in.resize(0,0);
        iteratoin_f_t.push_back(data_f_t);
        data_f_t.resize(0,0);
        iteratoin_i_t.push_back(data_i_t);
        data_i_t.resize(0,0);
        iteratoin_o_t.push_back(data_o_t);
        data_o_t.resize(0,0);
        iteratoin_concate.push_back(data_concate);
        data_concate.resize(0,0);
    }

    this->states = iteratoin_state;
    this->C_t_storage = iteratoin_C_t;
    this->C_in_storage = iteratoin_C_in;
    this->f_t_storage = iteratoin_f_t;
    this->i_t_storage = iteratoin_i_t;
    this->o_t_storage = iteratoin_o_t;
    this->concate_h_x = iteratoin_concate;

    return rst_mat;
}

//backward function
void lstm_rnn::gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind){
    //input_size: [nData * nlength * nDim]
    //get the variables
    int nData = dLdy.size();
    int nLength = dLdy[0].rows();
    int cocate_dim = this->w_f.rows();
    int hidden_dim = this->w_f.cols();
    int output_dim = this->v_output.cols();
    //if the length doesn't math, throw exception
    if(nLength!=y_ind.size()){
        throw std::invalid_argument("The input index must have the same length as the number of backpropagated data!");
    }
    MatrixXmat temp_dLdw_f = MatrixXmat::Zero(cocate_dim,hidden_dim);
    MatrixXmat temp_dLdb_f = MatrixXmat::Zero(1,hidden_dim);
    MatrixXmat temp_dLdw_i = MatrixXmat::Zero(cocate_dim,hidden_dim);
    MatrixXmat temp_dLdb_i = MatrixXmat::Zero(1,hidden_dim);
    MatrixXmat temp_dLdw_c = MatrixXmat::Zero(cocate_dim,hidden_dim);
    MatrixXmat temp_dLdb_c = MatrixXmat::Zero(1,hidden_dim);
    MatrixXmat temp_dLdw_o = MatrixXmat::Zero(cocate_dim,hidden_dim);
    MatrixXmat temp_dLdb_o = MatrixXmat::Zero(1,hidden_dim);
    MatrixXmat temp_dLdv = MatrixXmat::Zero(hidden_dim,output_dim);
    MatrixXmat current_dLdy;
    MatrixXmat current_dydv;
    for(unsigned int i=0;i<nLength;i++){
        int t = y_ind[i];
        //cout<<"zjr"<<endl;
        current_dLdy = ind_data_exact(dLdy,i+1);
        current_dydv = ind_data_exact(this->states,t).transpose();
        //cout<<"a"<<endl;
        //forget gate
        temp_dLdw_f = temp_dLdw_f + this->get_para_gradient(current_dLdy,"w_f",t);
        temp_dLdb_f = temp_dLdb_f + this->get_para_gradient(current_dLdy,"b_f",t);
        //cout<<"m"<<endl;
        //input gate i
        temp_dLdw_i = temp_dLdw_i + this->get_para_gradient(current_dLdy,"w_i",t);
        temp_dLdb_i = temp_dLdb_i + this->get_para_gradient(current_dLdy,"b_i",t);
        //cout<<"b"<<endl;
        //input gate c
        temp_dLdw_c = temp_dLdw_c + this->get_para_gradient(current_dLdy,"w_c",t);
        temp_dLdb_c = temp_dLdb_c + this->get_para_gradient(current_dLdy,"b_c",t);
        //cout<<"e"<<endl;
        //output gate
        temp_dLdw_o = temp_dLdw_o + this->get_para_gradient(current_dLdy,"w_o",t);
        temp_dLdb_o = temp_dLdb_o + this->get_para_gradient(current_dLdy,"b_o",t);
        //cout<<"r"<<endl;
        //output transformation
        temp_dLdv = temp_dLdv + (current_dydv*current_dLdy);
        //cout<<"ccc"<<endl;
    }

    this->dLdw_f = temp_dLdw_f;
    this->dLdb_f = temp_dLdb_f;
    this->dLdw_i = temp_dLdw_i;
    this->dLdb_i = temp_dLdb_i;
    this->dLdw_c = temp_dLdw_c;
    this->dLdb_c = temp_dLdb_c;
    this->dLdw_o = temp_dLdw_o;
    this->dLdb_o = temp_dLdb_o;
    this->dLdv = temp_dLdv;
}

MatrixXmat lstm_rnn::get_para_gradient(MatrixXmat dLdy, string variable, int start_time){
    //input: [nData * n_out_dim] dLdy matrix
    //output: dLd(.), the gradient of the current time-index back propagation
    int nLength = this->x[0].rows();
    int input_dim = this->w_input.rows();
    int hidden_dim = this->u_tranform.rows();
    int concate_dim =  this->w_f.rows();
    int output_dim = this->v_output.cols();
    //declare the variables of the gradient matrix
    int gradient_mat_width;
    int gradient_mat_length;
    if(variable=="w_f"||variable=="w_i"||variable=="w_c"||variable=="w_o"){
        gradient_mat_width = concate_dim;
        gradient_mat_length = hidden_dim;
    }
    else if(variable=="b_f"||variable=="b_i"||variable=="b_c"||variable=="b_o"){
        gradient_mat_width = 1;
        gradient_mat_length = hidden_dim;
    }
    MatrixXmat dLd_theta = MatrixXmat::Zero(gradient_mat_width,gradient_mat_length);
    MatrixXmat current_dLd_theta;
    MatrixXmat temp_mat;
    for(unsigned int t=start_time;t>0;t--){
        int ite_time = start_time-t;
        current_dLd_theta = this->dLdtheta_compute(t,variable);
        temp_mat = current_dLd_theta*dLdy*((this->v_output).transpose());
        temp_mat = recursive_func(temp_mat,ite_time,t);
        dLd_theta = dLd_theta + temp_mat;
    }

    return dLd_theta;
}

MatrixXmat lstm_rnn::recursive_func(MatrixXmat mat, int ite_time, int t){
    int m_amount = this->x.size();
    int input_dim = this->w_input.rows();
    int hidden_dim = this->u_tranform.rows();
    int concate_dim =  this->w_f.rows();
    int output_dim = this->v_output.cols();
    int mat_height = mat.rows();
    int mat_width = mat.cols();
    //extract the matrix
    MatrixXmat this_concate = ind_data_exact(this->concate_h_x,t);
    MatrixXmat this_o_t = ind_data_exact(this->o_t_storage,t);
    MatrixXmat this_c_t = ind_data_exact(this->C_t_storage,t);
    MatrixXmat this_c_t_1;
    MatrixXmat this_f_t = ind_data_exact(this->f_t_storage,t);
    MatrixXmat this_i_t = ind_data_exact(this->i_t_storage,t);
    MatrixXmat this_c_in = ind_data_exact(this->C_in_storage,t);
    int n_hidden = this_o_t.cols();
    if(t>1){
        this_c_t_1 = ind_data_exact(this->C_t_storage,t-1);
    }
    else{
        this_c_t_1 = MatrixXmat::Zero(m_amount,n_hidden);
    }
    //compute the dHdh here -- not very easy....
    MatrixXmat rst_mat;
    //get the submatrix of the weight
    MatrixXmat w_f_h = mat_extract(this->w_f,1,hidden_dim,1,hidden_dim);
    MatrixXmat w_i_h = mat_extract(this->w_i,1,hidden_dim,1,hidden_dim);
    MatrixXmat w_c_h = mat_extract(this->w_c,1,hidden_dim,1,hidden_dim);
    MatrixXmat w_o_h = mat_extract(this->w_o,1,hidden_dim,1,hidden_dim);
    //diagonal matrix to compute dHdh
    MatrixXmat para_w_o = MatrixXmat::Zero(hidden_dim, hidden_dim);
    MatrixXmat para_w_f = MatrixXmat::Zero(hidden_dim, hidden_dim);
    MatrixXmat para_w_i = MatrixXmat::Zero(hidden_dim, hidden_dim);
    MatrixXmat para_w_c = MatrixXmat::Zero(hidden_dim, hidden_dim);
    MatrixXmat para_w_f_c_i = MatrixXmat::Zero(hidden_dim, hidden_dim);
    //temp mats to help computation
    MatrixXmat o_t_data;
    MatrixXmat c_t_data;
    MatrixXmat c_t_1_data;
    MatrixXmat f_t_data;
    MatrixXmat i_t_data;
    MatrixXmat c_in_data;
    for(unsigned int i=0;i<m_amount;i++){
        o_t_data = mat_extract(this_o_t,i+1,i+1,1,hidden_dim);
        c_t_data = mat_extract(this_c_t,i+1,i+1,1,hidden_dim);
        c_t_1_data = mat_extract(this_c_t_1,i+1,i+1,1,hidden_dim);
        f_t_data = mat_extract(this_f_t,i+1,i+1,1,hidden_dim);
        i_t_data = mat_extract(this_i_t,i+1,i+1,1,hidden_dim);
        c_in_data = mat_extract(this_c_in,i+1,i+1,1,hidden_dim);
        para_w_o = para_w_o + mat_diagnolize((tanh_func(c_t_data).cwiseProduct(\
                                    o_t_data)).cwiseProduct(MatrixXmat::Ones(1,hidden_dim) - o_t_data));
        para_w_f = para_w_f + mat_diagnolize((c_t_1_data.cwiseProduct(\
                                    f_t_data)).cwiseProduct(MatrixXmat::Ones(1,hidden_dim) - f_t_data));
        para_w_i = para_w_i + mat_diagnolize((c_in_data.cwiseProduct(\
                                    i_t_data)).cwiseProduct(MatrixXmat::Ones(1,hidden_dim) - i_t_data));
        para_w_c = para_w_c + mat_diagnolize((MatrixXmat::Ones(1,hidden_dim)\
                                - c_in_data.cwiseProduct(c_in_data)).cwiseProduct(i_t_data));
        para_w_f_c_i = para_w_f_c_i + mat_diagnolize((MatrixXmat::Ones(1,hidden_dim),\
                               - tanh_func(c_t_data).cwiseProduct(tanh_func(c_t_data))).cwiseProduct(o_t_data));
    }

    MatrixXmat dHdh_mat = para_w_o*(w_o_h.transpose()) + para_w_f_c_i*((para_w_f*(w_f_h.transpose())\
                                + para_w_c*(w_c_h.transpose())) + para_w_i*(para_w_i.transpose()));
    if(ite_time==0){
        rst_mat = mat;
    }
    else if(ite_time==1){
        rst_mat = mat*dHdh_mat;
    }
    else if(ite_time>1){
        rst_mat = recursive_func(mat,ite_time-1,t)*dHdh_mat;
    }
    else{
        throw std::invalid_argument("The input iteratoin time is invalid!");
    }

    return rst_mat;
}

MatrixXmat lstm_rnn::dLdtheta_compute(int t, string mode){
    //compute dhd_theta here
    MatrixXmat this_concate = ind_data_exact(this->concate_h_x,t);
    MatrixXmat this_o_t = ind_data_exact(this->o_t_storage,t);
    MatrixXmat this_c_t = ind_data_exact(this->C_t_storage,t);
    MatrixXmat this_c_t_1;
    MatrixXmat this_f_t = ind_data_exact(this->f_t_storage,t);
    MatrixXmat this_i_t = ind_data_exact(this->i_t_storage,t);
    MatrixXmat this_c_in = ind_data_exact(this->C_in_storage,t);
    int m_amount = this_concate.rows();
    int concate_len = this_concate.cols();
    int n_hidden = this_o_t.cols();
    if(t>1){
        this_c_t_1 = ind_data_exact(this->C_t_storage,t-1);
    }
    else{
        this_c_t_1 = MatrixXmat::Zero(m_amount,n_hidden);
    }
    MatrixXmat para_mat;
    MatrixXmat tan_derivetive = MatrixXmat::Ones(m_amount,n_hidden) - tanh_func(this_c_t).cwiseProduct(tanh_func(this_c_t));
    if(mode=="w_f"){
        para_mat = (((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_c_t_1)).cwiseProduct(this_f_t)).cwiseProduct(\
                    (MatrixXmat::Ones(m_amount,n_hidden) - this_f_t));
    }
    else if(mode=="b_f"){
        para_mat =(((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_c_t_1)).cwiseProduct(this_f_t)).cwiseProduct(\
                                (MatrixXmat::Ones(m_amount,n_hidden) - this_f_t));
    }
    else if(mode=="w_i"){
        para_mat = (((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_c_in)).cwiseProduct(\
                                this_i_t)).cwiseProduct((MatrixXmat::Ones(m_amount,n_hidden) - this_i_t));
    }
    else if(mode=="b_i"){
        para_mat = (((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_c_in)).cwiseProduct(\
                                this_i_t)).cwiseProduct((MatrixXmat::Ones(m_amount,n_hidden) - this_i_t));
    }
    else if(mode=="w_c"){
        para_mat = ((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_i_t)).cwiseProduct(\
                                (MatrixXmat::Ones(m_amount,n_hidden)-(this_c_in.cwiseProduct(this_c_in))));
    }
    else if(mode=="b_c"){
        para_mat = ((this_o_t.cwiseProduct(tan_derivetive)).cwiseProduct(this_i_t)).cwiseProduct(\
                                (MatrixXmat::Ones(m_amount,n_hidden)-(this_c_in.cwiseProduct(this_c_in))));
    }
    else if(mode=="w_o"){
        para_mat = (tanh_func(this_c_t).cwiseProduct(this_o_t)).cwiseProduct(MatrixXmat::Ones(m_amount,n_hidden)-this_o_t);
    }
    else if(mode=="b_o"){
        para_mat = (tanh_func(this_c_t).cwiseProduct(this_o_t)).cwiseProduct(MatrixXmat::Ones(m_amount,n_hidden)-this_o_t);
    }
    else{
        throw std::invalid_argument("Gradient compute mode unrecognized!");
    }
    //compute the parameter matrix data-by-data
    MatrixXmat rst_mat;
    if(mode=="w_f"||mode=="w_i"||mode=="w_c"||mode=="w_o"){
        rst_mat = MatrixXmat::Zero(m_amount,concate_len);
    }
    else if(mode=="b_f"||mode=="b_i"||mode=="b_c"||mode=="b_o"){
        rst_mat = MatrixXmat::Zero(m_amount,1);
    }
    else{
        throw std::invalid_argument("Gradient compute mode unrecognized!");
    }
    MatrixXmat dup_concate_data;
    MatrixXmat current_para_mat;
    for(unsigned int i=0;i<m_amount;i++){
        current_para_mat = para_mat.row(i);
        if(mode=="w_f"||mode=="w_i"||mode=="w_c"||mode=="w_o"){
            dup_concate_data = this_concate.row(i).replicate(n_hidden,1);
        }
        else if(mode=="b_f"||mode=="b_i"||mode=="b_c"||mode=="b_o"){
            dup_concate_data = MatrixXmat::Ones(n_hidden,1);
        }
        else{
            throw std::invalid_argument("Gradient compute mode unrecognized!");
        }
        rst_mat.row(i) = current_para_mat*dup_concate_data;
    }
    rst_mat.transposeInPlace();

    return rst_mat;
}
