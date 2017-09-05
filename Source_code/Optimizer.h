#ifndef OPTIMIZER_H_INCLUDED
#define OPTIMIZER_H_INCLUDED
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"

using namespace std;
using namespace Eigen;

//basic optimizer, not used in practice but acting as a bisic class
template<typename decimal>
class basic_optimizer{
protected:
    int variable_amount;
    decimal learning_rate;
    MatrixXmat gradient_learn;
public:
    basic_optimizer(decimal learning_rate, int variable_amount){
        this->learning_rate = learning_rate;
        this->variable_amount = variable_amount;
    };
    ~basic_optimizer(){
    };
    MatrixXmat update_function(MatrixXmat para_mat){
        MatrixXmat rst_mat = para_mat-this->gradient_learn;

        return rst_mat;
    };
};

//Stochastic gradient descent
template<typename decimal>
class sgd_optimizer:public basic_optimizer<decimal>{
public:
    sgd_optimizer(decimal learning_rate, int variable_amount):basic_optimizer<decimal>(learning_rate,variable_amount){};
    ~sgd_optimizer(){};
    void get_learn_mat(MatrixXmat gradient, int variable_ind){
        this->gradient_learn = this->learning_rate*gradient;
    };
};

//Momentum
template<typename decimal>
class Momentum_optimizer:public basic_optimizer<decimal>{
protected:
    decimal gamma;
    vector<MatrixXmat> gradient_learn_prev;
public:
    Momentum_optimizer(decimal learning_rate, int variable_amount, decimal gamma):basic_optimizer<decimal>(learning_rate,variable_amount){
        this->gamma = gamma;
        vector<MatrixXmat> temp_mat(variable_amount);   //only initilize the first dimension, which means the number of variables
        this->gradient_learn_prev = temp_mat;
    };
    ~Momentum_optimizer(){};
    void get_learn_mat(MatrixXmat gradient, int variable_ind){
        if((this->gradient_learn_prev[variable_ind]).rows()==0){
            this->gradient_learn = this->learning_rate*gradient;
        }
        else{
            this->gradient_learn = this->learning_rate*this->gradient_learn + this->gamma*this->gradient_learn_prev[variable_ind];
        }
        this->gradient_learn_prev[variable_ind] = this->gradient_learn;
    };
};

//Adagrad
template<typename decimal>
class Adagrad_optimizer:public basic_optimizer<decimal>{
protected:
    decimal epsilon;
    //The G matrix of the Adagrad
    vector<MatrixXmat> G_t_mat;
public:
    Adagrad_optimizer(decimal learning_rate, int variable_amount, decimal epsilon):basic_optimizer<decimal>(learning_rate,variable_amount){
        this->epsilon = epsilon;
        vector<MatrixXmat> temp_mat(variable_amount);   //only initilize the first dimension, which means the number of variables
        this->G_t_mat = temp_mat;
    };
    ~Adagrad_optimizer(){};
    void get_learn_mat(MatrixXmat gradient, int variable_ind){
        int nDim = gradient.rows();
        int nNode = gradient.cols();
        MatrixXmat para_mat(nDim,nNode);
        if((this->G_t_mat[variable_ind]).rows()==0){
            //Get initial G matrix
            this->G_t_mat[variable_ind] = gradient.cwiseProduct(gradient);
        }
        else{
            //update G matrix
            this->G_t_mat[variable_ind] = this->G_t_mat[variable_ind]+gradient.cwiseProduct(gradient);
        }
        for(unsigned int j=0;j<nNode;j++){
            for(unsigned int i=0;i<nDim;i++){
                para_mat(i,j) = (this->learning_rate)/(UnitType)sqrt((double)(G_t_mat[variable_ind](i,j)+(this->epsilon)));
            }
        }
        //Get update matrix
        this->gradient_learn = gradient.cwiseProduct(para_mat);
        };
};

//Adadelta
template<typename decimal>
class Adadelta_optimizer:public basic_optimizer<decimal>{
protected:
    decimal epsilon;
    decimal gamma;
    //The G matrix of the Adagrad
    vector<MatrixXmat> G_t_mat;
public:
    Adadelta_optimizer(decimal learning_rate, int variable_amount, decimal epsilon, decimal gamma):basic_optimizer<decimal>(learning_rate,variable_amount){
        this->epsilon = epsilon;
        this->gamma = gamma;
        vector<MatrixXmat> temp_mat(variable_amount);   //only initilize the first dimension, which means the number of variables
        this->G_t_mat = temp_mat;
    };
    ~Adadelta_optimizer(){};
    void get_learn_mat(MatrixXmat gradient, int variable_ind){
        int nDim = gradient.rows();
        int nNode = gradient.cols();
        MatrixXmat para_mat(nDim,nNode);
        if((this->G_t_mat[variable_ind]).rows()==0){
            //Get initial G matrix
            this->G_t_mat[variable_ind] = (1-(this->gamma))*gradient.cwiseProduct(gradient);
        }
        else{
            //update G matrix
            this->G_t_mat[variable_ind] = this->gamma*this->G_t_mat[variable_ind] + (1-(this->gamma))*gradient.cwiseProduct(gradient);
        }
        for(unsigned int j=0;j<nNode;j++){
            for(unsigned int i=0;i<nDim;i++){
                para_mat(i,j) = (this->learning_rate)/(UnitType)sqrt((double)(G_t_mat[variable_ind](i,j)+(this->epsilon)));
            }
        }
        //Get update matrix
        this->gradient_learn = gradient.cwiseProduct(para_mat);
    };
};

//Adam optimizer
template<typename decimal>
class Adam_optimizer:public basic_optimizer<decimal>{
protected:
    decimal beta_1;
    decimal beta_2;
    decimal epsilon;
    //first-order moment
    vector<MatrixXmat> m_t_mat;
    //second-order moment
    vector<MatrixXmat> v_t_mat;
public:
    Adam_optimizer(decimal learning_rate, int variable_amount, decimal beta_1, decimal beta_2 ,decimal epsilon):basic_optimizer<decimal>(learning_rate,variable_amount){
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        vector<MatrixXmat> temp_mat(variable_amount);   //only initilize the first dimension, which means the number of variables
        this->m_t_mat = temp_mat;
        this->v_t_mat = temp_mat;
    };
    ~Adam_optimizer(){};
    void get_learn_mat(MatrixXmat gradient, int variable_ind){
        //get the shape
        int nDim = gradient.rows();
        int nNode = gradient.cols();
        //construct update matrix
        MatrixXmat m_update_mat;
        MatrixXmat v_update_mat;
        //para_mat
        MatrixXmat para_mat(nDim,nNode);
        if((this->m_t_mat[variable_ind]).rows()==0){
            //Get initial G matrix
            this->m_t_mat[variable_ind] = (1-(this->beta_1))*gradient;
            this->v_t_mat[variable_ind] = (1-(this->beta_2))*gradient.cwiseProduct(gradient);  //[nDim+1 * nNode]
        }
        else{
            //update moment matrix
            this->m_t_mat[variable_ind] = (this->beta_1)*this->m_t_mat[variable_ind] + (1-(this->beta_1))*gradient;

            this->v_t_mat[variable_ind] = this->beta_2*this->v_t_mat[variable_ind] + (1-(this->beta_2))*gradient.cwiseProduct(gradient);
        }
        //Compute m_update and v_update
        //m matrix
        m_update_mat = (UnitType)(1/(1-(this->beta_1)))*this->m_t_mat[variable_ind];
        //v matrix
        v_update_mat = (UnitType)(1/(1-(this->beta_2)))*this->v_t_mat[variable_ind];
        //compute parameter matrix
        for(unsigned int j=0;j<nNode;j++){
            for(unsigned int i=0;i<nDim;i++){
                para_mat(i,j) = (this->learning_rate)/(UnitType)sqrt((double)(v_update_mat(i,j)+(this->epsilon)));
            }
        }
        //Get update matrix
        this->gradient_learn = m_update_mat.cwiseProduct(para_mat);
    };
};
#endif // OPTIMIZER_H_INCLUDED
