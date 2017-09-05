#ifndef RNN_H_INCLUDED
#define RNN_H_INCLUDED
#include "Optimizer.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include <vector>

using namespace std;
using namespace Eigen;

MatrixXmat ouput_reshape_rnn(vector<MatrixXmat> data);
vector<MatrixXmat> input_reshape_rnn(MatrixXmat data, int nLength, int nDim);
MatrixXmat ind_data_exact(vector<MatrixXmat> data, int index);

namespace RNN{
//base class for all the RNN classes
class Recurrent_net{
protected:
    //n1: dimension of data input n2: dimension of rnn size  n3: Dimension of the output
    int input_Dim;   //The dimension of input data
    int rnn_Dim;     //The dimension of the rnn
    int output_Dim;
    int rnn_length;
    //store the data
    vector<MatrixXmat> x;
    vector<MatrixXmat> states;
    //store the state
    MatrixXmat rnn_state;
    //forward tranformation matrix
    MatrixXmat w_input;            //[n1*n2] vector
    MatrixXmat u_tranform;         //[n2*n2] vector
    MatrixXmat v_output;           //[n2*n3] vector
    //gradient matrix
    MatrixXmat dLdw;             //[n1*n2] vector
    MatrixXmat dLdu;             //[n2*n2] vector
    MatrixXmat dLdv;            //[n2*n3] vector
public:
    Recurrent_net(int input_Dim, int rnn_Dim, int output_Dim);
    ~Recurrent_net();
    void w_initialize();
    //compute the gradients of a specific parameter
    virtual MatrixXmat get_para_gradient(MatrixXmat dLdy, string variable, int start_ind);
    //compute the gradients of all the parameters
    virtual void gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind);
    //function that multiply the matrix recursively
    virtual MatrixXmat recursive_func(MatrixXmat mat, int ite_time);
    //function to compute dldtheta
    virtual MatrixXmat dLdtheta_compute(int t, string mode);
};

//basic RNN cell
class basic_rnn:public Recurrent_net{
public:
    basic_rnn(int input_Dim, int rnn_Dim, int output_Dim): Recurrent_net(input_Dim,rnn_Dim,output_Dim){};
    ~basic_rnn();
    //function to store information
    vector<MatrixXmat> Para_copy_store(){
        vector<MatrixXmat> para_mat(3);
        para_mat[0] = this->w_input;
        para_mat[1] = this->u_tranform;
        para_mat[2] = this->v_output;

        return para_mat;
    }
    void Para_copy_read(vector<MatrixXmat> para_mat){
        this->w_input = para_mat[0];
        this->u_tranform = para_mat[1];
        this->v_output = para_mat[2];
    }
    //forward function
    vector<MatrixXmat> forward_func(vector<MatrixXmat> x);
    //compute the gradients of all the parameters
    void gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind);
    MatrixXmat get_para_gradient(MatrixXmat dLdy, string variable, int start_ind);
    //recursive function of the basic rnn
    MatrixXmat recursive_func(MatrixXmat mat, int ite_time);
    //function to compute dldtheta
    MatrixXmat dLdtheta_compute(int t, string mode);
    //the training function
    template<typename Opt>
    void train(Opt& optimizer){
        //update w mattrix (affulix matrix 0)
        optimizer.get_learn_mat(this->dLdw,0);
        w_input = optimizer.update_function(w_input);
        //update u matrix (affulix matrix 1)
        optimizer.get_learn_mat(this->dLdu,1);
        u_tranform = optimizer.update_function(u_tranform);
        //udate v matrix (affulix matrix 2)
        optimizer.get_learn_mat(this->dLdv,2);
        v_output = optimizer.update_function(v_output);
    }
};

class lstm_rnn:public Recurrent_net{
protected:
    vector<MatrixXmat> C_t_storage;    //C_t
    vector<MatrixXmat> C_in_storage;
    vector<MatrixXmat> f_t_storage;    //forget input
    vector<MatrixXmat> i_t_storage;    //i input
    vector<MatrixXmat> o_t_storage;    //o output
    vector<MatrixXmat> concate_h_x;
    //matrix for forget gate
    MatrixXmat w_f;
    MatrixXmat b_f;
    //matrix for state gates
    MatrixXmat w_i;
    MatrixXmat b_i;
    MatrixXmat w_c;
    MatrixXmat b_c;
    //matrix for output gate
    MatrixXmat w_o;
    MatrixXmat b_o;
    //gradient parameters
    MatrixXmat dLdw_f;
    MatrixXmat dLdb_f;
    MatrixXmat dLdw_i;
    MatrixXmat dLdb_i;
    MatrixXmat dLdw_c;
    MatrixXmat dLdb_c;
    MatrixXmat dLdw_o;
    MatrixXmat dLdb_o;
public:
    lstm_rnn(int input_Dim, int rnn_Dim, int output_Dim): Recurrent_net(input_Dim,rnn_Dim,output_Dim){
        this->w_initialize();
    };
    ~lstm_rnn();
    void w_initialize();
    vector<MatrixXmat> forward_func(vector<MatrixXmat> x);
    void gradient_func(vector<MatrixXmat> dLdy, vector<int> y_ind);
    MatrixXmat get_para_gradient(MatrixXmat dLdy, string variable, int start_ind);
    //recursive function of the basic rnn
    MatrixXmat recursive_func(MatrixXmat mat, int ite_time, int t);
    //function to compute dldtheta
    MatrixXmat dLdtheta_compute(int t, string mode);
    template <typename Opt>
    void train(Opt& optimizer){
        //update w mattrix (affulix matrix 0)
        optimizer.get_learn_mat(this->dLdv,0);
        v_output = optimizer.update_function(v_output);
        //update w_f matrix (affulix matrix 1)
        optimizer.get_learn_mat(this->dLdw_f,1);
        w_f = optimizer.update_function(w_f);
        //udate b_f matrix (affulix matrix 2)
        optimizer.get_learn_mat(this->dLdb_f,2);
        b_f = optimizer.update_function(b_f);
        //udate w_i matrix (affulix matrix 3)
        optimizer.get_learn_mat(this->dLdw_i,3);
        w_i = optimizer.update_function(w_i);
        //udate b_i matrix (affulix matrix 4)
        optimizer.get_learn_mat(this->dLdb_i,4);
        b_i = optimizer.update_function(b_i);
        //udate w_c matrix (affulix matrix 5)
        optimizer.get_learn_mat(this->dLdw_c,5);
        w_c = optimizer.update_function(w_c);
        //udate b_c matrix (affulix matrix 6)
        optimizer.get_learn_mat(this->dLdb_c,6);
        b_c = optimizer.update_function(b_c);
        //udate w_o matrix (affulix matrix 7)
        optimizer.get_learn_mat(this->dLdw_o,7);
        w_o = optimizer.update_function(w_o);
        //udate b_o matrix (affulix matrix 8)
        optimizer.get_learn_mat(this->dLdb_o,8);
        b_o = optimizer.update_function(b_o);
    }
};
};

#endif // RNN_H_INCLUDED
