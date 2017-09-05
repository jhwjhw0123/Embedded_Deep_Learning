#ifndef CNN_H_INCLUDED
#define CNN_H_INCLUDED
#include "Optimizer.h"
#include "MatrixOperation.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include <vector>

using namespace Eigen;

vector<vector<MatrixXmat> > data_padding(vector<vector<MatrixXmat> > x, int padding_height, int padding_width);
//Special functions for CNN to perform forward/backward propagation
vector<vector<MatrixXmat> > data_squaring(MatrixXmat x, int height, int width, int channel);
MatrixXmat data_flatting(vector<vector<MatrixXmat> > x);

namespace CNN{

struct pooling_information{
    UnitType pooling_value;
    MatrixXmat this_dydx;
};

class Convolution_layer{
protected:
    int input_channel;
    int output_channel;
    int kernel_size;
    int stride;
    int padding_height;
    int padding_width;
    string mode;
    vector<vector<MatrixXmat> > conv_weights;  //should be [input_channel * output_channel * conv_height * conv_width]
    vector<vector<MatrixXmat> > conv_gradient;  // should be [input_channel * output_channel * conv_height * conv_width]
    vector<vector<MatrixXmat> > x;               //[nData * nChannels * images(2-dims)]
    vector<vector<MatrixXmat> > padded_x;
    vector<vector<MatrixXmat> > dLdx;
public:
    vector<vector<MatrixXmat> > conv_weights_copy;
    //function for storing and reading parameter
    vector<vector<MatrixXmat> > Para_copy_store(){
        return conv_weights;
    }
    void Para_copy_read(vector<vector<MatrixXmat> > conv_weight_read){
        this->conv_weights = conv_weight_read;
    }
    Convolution_layer(int input_channel,int output_channel, int kernel_size, int stride, string mode);
    ~Convolution_layer();
    void weight_initialize();
    vector<vector<MatrixXmat> > forward_func(vector<vector<MatrixXmat> > x);
    vector<vector<MatrixXmat> > forward_func_fast(vector<vector<MatrixXmat> > x);
    vector<vector<MatrixXmat> > backward_func(vector<vector<MatrixXmat> > dLdy);
    template <typename Opt>
    void train(Opt &optimizer, string regulization, UnitType lambda){
        int nheight = this -> conv_gradient[0][0].rows();
        int nwidth = this -> conv_gradient[0][0].cols();
        UnitType threshold = 0.01;
        for(unsigned int i=0;i<this->input_channel;i++){
            for(unsigned int j=0;j<this->output_channel;j++){
                MatrixXmat reg_mat = MatrixXmat::Zero(nheight,nwidth);
                MatrixXi max_ind;
                if(regulization=="None"){
                }
                else if(regulization=="l_1"){
                    for(unsigned int k=0;k<nheight;k++){
                        for(unsigned int p=0;p<nwidth;p++){
                            if(abs(this->conv_weights[i][j](k,p))<threshold){
                                reg_mat(k,p) = 0;
                            }
                            else if(this->conv_weights[i][j](k,p)>0){
                                reg_mat(k,p) = lambda*1;
                            }
                            else{
                                reg_mat(k,p) = lambda*(-1);
                            }
                        }
                    }
                }
                else if(regulization=="l_2"){
                    reg_mat = lambda*this->conv_weights[i][j];
                }
                else if(regulization=="l_inf"){
                    max_ind = mat_argmax(this->conv_weights[i][j],"row");
                    for(unsigned int k=0;k<nheight;k++){
                        int this_ind = max_ind(k,0);
                        reg_mat(k,this_ind) = lambda*this->conv_weights[i][j](k,this_ind);
                    }
                }
                MatrixXmat learn_mat = this->conv_gradient[i][j] + reg_mat;
                optimizer.get_learn_mat(learn_mat,(i*this->output_channel)+j);
                this->conv_weights[i][j] = optimizer.update_function(this->conv_weights[i][j]);
            }
        }
    }
};

//basic pooling layer class
class pooling_layer{
protected:
    int kernel_size;
    int stride;
    int padding_height;
    int padding_width;
    string overlap_mode;
    string padding_mode;
    vector<vector<MatrixXmat> > x;
    vector<vector<MatrixXmat> > padded_x;
    vector<vector<vector<vector<MatrixXmat> > > > dydx;   //store dydx to perform backword [nData*nChannel*new_height*new_width*k_height*k+width]
public:
    pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode);
    ~pooling_layer();
    vector<vector<MatrixXmat> > forward_func(vector<vector<MatrixXmat> > x);
    vector<vector<MatrixXmat> > backward_func(vector<vector<MatrixXmat> > dLdy);
    virtual pooling_information get_pooling(MatrixXmat current_image);
};

//max-pooling class
class max_pooling_layer:public pooling_layer{
public:
    max_pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode):pooling_layer(kernel_size,stride,overlap_mode,padding_mode){};
    ~max_pooling_layer();
    pooling_information get_pooling(MatrixXmat current_image);
};

//sum-pooling class
class sum_pooling_layer:public pooling_layer{
public:
    sum_pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode):pooling_layer(kernel_size,stride,overlap_mode,padding_mode){};
    ~sum_pooling_layer();
    pooling_information get_pooling(MatrixXmat current_image);
};

//average-pooling class
class average_pooling_layer:public pooling_layer{
public:
    average_pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode):pooling_layer(kernel_size,stride,overlap_mode,padding_mode){};
    ~average_pooling_layer();
    pooling_information get_pooling(MatrixXmat current_image);
};

//stochastic-pooling (this is an equivalent of dropout in CNN)
class stochastic_pooling_layer: public pooling_layer{
public:
    stochastic_pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode):pooling_layer(kernel_size,stride,overlap_mode,padding_mode){};
    ~stochastic_pooling_layer();
    pooling_information get_pooling(MatrixXmat current_image);
};

};

#endif // CNN_H_INCLUDED
