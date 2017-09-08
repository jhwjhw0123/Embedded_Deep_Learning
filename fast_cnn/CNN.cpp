#include <cmath>
#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdexcept>
#include "CNN.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MathType.h"
#include "MatrixOperation.h"
#include "StatisticMath.h"

using namespace std;
using namespace CNN;
using namespace Eigen;

/******************padding routine********************/
vector<vector<MatrixXmat> > data_padding(vector<vector<MatrixXmat> > x, int padding_height, int padding_width){
    //data [nData * channels * height * width]
    int nData = x.size();
    int input_channel = x[0].size();
    int height = x[0][0].rows();
    int width = x[0][0].cols();
    //result mat
    vector<vector<MatrixXmat> > rst_mat(nData,vector<MatrixXmat>(input_channel));
    //vector to store the image
    MatrixXmat current_image(height,width);
    //height padding
    MatrixXmat height_padding = MatrixXmat::Zero(padding_height,width);
    MatrixXmat width_padding = MatrixXmat::Zero(height+(2*padding_height),padding_width);
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<input_channel;j++){
            current_image = x[i][j];
            if(padding_height!=0){
                //top padding
                current_image = mat_concate(height_padding,current_image,0);
                //bottom padding
                current_image = mat_concate(current_image,height_padding,0);
            }
            if(padding_width!=0){
                //left padding
                current_image = mat_concate(width_padding,current_image,1);
                //right padding
                current_image = mat_concate(current_image,width_padding,1);
            }
            //push the image to result matrix
            rst_mat[i][j] = current_image;
        }
    }

    return rst_mat;
}

/**************Tranfer data into Image-like***************/
vector<vector<MatrixXmat> > data_squaring(MatrixXmat x, int height, int width, int channel){
    //input should be [nData * nDim]
    //output should be [nData * channel * height * width]
    int nData = x.rows();
    int nDim = x.cols();
    //construct the vectors of transformed data
    vector<vector<MatrixXmat> > squared_data(nData,vector<MatrixXmat>(channel));
    MatrixXmat temp_image;
    if(nDim%channel!=0){
        throw std::invalid_argument("The input dimension amount must be divided by the number of channels!");
    }
    if(nDim!=height*width*channel){
        throw std::invalid_argument("The dimension and image shape must match!");
    }
    int data_len = nDim/channel;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<channel;j++){
            int this_data_ind = i+1;
            int start_column = j*data_len+1;
            int end_column = (j+1)*data_len;
            temp_image = mat_extract(x,this_data_ind,this_data_ind,start_column,end_column);
            squared_data[i][j] = mat_reshape(temp_image,height,width,1);
        }
    }

    return squared_data;
}

/**************Transfer data into flatten form [nData*nDim]****************/
MatrixXmat data_flatting(vector<vector<MatrixXmat> > x){
    //input:[nData * nChannel * height * width]
    int nData = x.size();
    int nChannel = x[0].size();
    int height = x[0][0].rows();
    int width = x[0][0].cols();
    int nDim = nChannel*height*width;
    MatrixXmat flat_data = MatrixXmat::Zero(nData,nDim);
    //push the data as [channel -> height -> width]
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nChannel;j++){
           for(unsigned int k=0;k<height;k++){
            for(unsigned int p=0;p<width;p++){
                int dim_ind = j*height*width + k*width + p;
                flat_data(i,dim_ind) = x[i][j](k,p);
            }
           }
        }
    }

    return flat_data;
}


/**********Convolutional Layer Codes************/
Convolution_layer::Convolution_layer(int input_channel,int output_channel, int kernel_size, int stride, string mode){
    this->input_channel = input_channel;
    this->output_channel = output_channel;
    this->mode = mode;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->weight_initialize();
}

Convolution_layer::~Convolution_layer(){

}

void Convolution_layer::weight_initialize(){
    vector<vector<MatrixXmat> > temp_mat(this->input_channel, vector<MatrixXmat>(this->output_channel,\
                                                            MatrixXmat(this->kernel_size, this->kernel_size)));
    this->conv_weights = temp_mat;
    for(unsigned int i=0;i<input_channel;i++){
        for(unsigned int j=0;j<output_channel;j++){
            for(unsigned int k=0;k<kernel_size;k++){
                for(unsigned int p=0;p<kernel_size;p++){
                    this->conv_weights[i][j](k,p) = 0.01*normal_dist_gen_1d<UnitType>(0,1);
                }
            }
        }
    }
}

vector<vector<MatrixXmat> > Convolution_layer::forward_func(vector<vector<MatrixXmat> > x){
    //input data size [nData * channels * [height * width]]
    int nData = x.size();
    int height = x[0][0].rows();
    int width = x[0][0].cols();
    this->x = x;
    //depend to padding
    int padding_height = 0;
    int padding_width = 0;
    if(this->mode=="same"){
        padding_width = (this->kernel_size + ((this->stride)-1)*width - this->stride)/2;
        padding_height = (this->kernel_size + ((this->stride)-1)*height - this->stride)/2;
    }
    this->padding_height = padding_height;
    this->padding_width = padding_width;
    //Add 0-padding to the data
    vector<vector<MatrixXmat> > padded_x = data_padding(x,padding_height,padding_width);
    this->padded_x = padded_x;
    //Get the new shape of the convolved data
    int new_height = (int)((height+2*padding_height-(this->kernel_size))/this->stride) + 1;
    int new_width = (int)((width+2*padding_width-(this->kernel_size))/this->stride) + 1;
    vector<vector<MatrixXmat> > convolution_data(nData,vector<MatrixXmat>(this->output_channel,\
                                                    MatrixXmat::Zero(new_height,new_width)));
    MatrixXmat current_input_channel;
    MatrixXmat convolution_window_data;
    MatrixXmat convolution_weight;
    MatrixXmat pixel_dot_mat;
    UnitType this_pixel_value;
    //determine the strding length
    int height_stride_len = height+2*padding_height - kernel_size + 1;
    int width_stride_len = width+2*padding_width - kernel_size + 1;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<this->output_channel;j++){
            //convolution_window_data.clear();
            //convolution_weight.clear();
            for(unsigned int k=0;k<height_stride_len;k+=this->stride){
                for(unsigned int p=0;p<width_stride_len;p+=this->stride){
                    this_pixel_value = 0;
                    for(unsigned int l=0;l<this->input_channel;l++){
                        //The weight of the lth input channel
                        current_input_channel = padded_x[i][l];
                        convolution_window_data = mat_extract(current_input_channel,k+1,k+kernel_size,p+1,p+kernel_size);
                        convolution_weight = this->conv_weights[l][j];
                        pixel_dot_mat = convolution_window_data.cwiseProduct(convolution_weight);
                        //assign the convolution result of this input channel
                        convolution_data[i][j](k,p) += pixel_dot_mat.sum();
                        //cout<<"nData: "<<i<<" Input Channel: "<<l<<" Output Channel: "<<j<<" row_start: "<<k<<" column_start: "<<p<<endl;
                        }
                }
            }
        }
    }

    return convolution_data;
}

//Fast convulutional forward function to process data
vector<vector<MatrixXmat> > Convolution_layer::forward_func_fast(vector<vector<MatrixXmat> > x){
    //Input: [nData * nInputChannel * nHeight * nWidth]
    //Output: [nData * nOutputChannel * nHeight_new * nWidth_new]
    int nData = x.size();
    int height = x[0][0].rows();
    int width = x[0][0].cols();
    this->x = x;
    //depend to padding
    int padding_height = 0;
    int padding_width = 0;
    if(this->mode=="same"){
        padding_width = (this->kernel_size + ((this->stride)-1)*width - this->stride)/2;
        padding_height = (this->kernel_size + ((this->stride)-1)*height - this->stride)/2;
    }
    this->padding_height = padding_height;
    this->padding_width = padding_width;
    //Add 0-padding to the data
    vector<vector<MatrixXmat> > padded_x = data_padding(x,padding_height,padding_width);
    this->padded_x = padded_x;
    //Get the new shape of the convolved data
    int new_height = (int)((height+2*padding_height-(this->kernel_size))/this->stride) + 1;
    int new_width = (int)((width+2*padding_width-(this->kernel_size))/this->stride) + 1;
    int new_entries = new_height*new_width;
    vector<vector<MatrixXmat> > convolution_data(nData,vector<MatrixXmat>(this->output_channel,\
                                                    MatrixXmat::Zero(new_height,new_width)));
    //Define the reshaped input
    vector<MatrixXmat> reshaped_padded_input_channels(this->input_channel,MatrixXmat(nData*new_entries,this->kernel_size*this->kernel_size));
    //Define the reshaped weights
    vector<MatrixXmat> reshaped_weights_channels(this->input_channel,MatrixXmat(this->kernel_size*this->kernel_size,this->output_channel));
    //Define other Matrix to temporarily store the image
    MatrixXmat current_input_channel;
    MatrixXmat convolution_window_data;
    MatrixXmat this_convolution_streched;
    MatrixXmat this_convolution_weight;
    MatrixXmat this_convolution_weight_streched;
    //determine the strding length
    int height_stride_len = height+2*padding_height - kernel_size + 1;
    int width_stride_len = width+2*padding_width - kernel_size + 1;
    //We drop the routine of looping the data here, instead use matrix multiplication method
    /**********Things Happening**********
    *1. Data is strided and reshaped into [(nData*nstrides),(nKernel_size * nInput_channel)]
    *2. Weights are reshaped into [(nKernel_size*n_inputchannel),nOuput_channel]
    *3. Directly perform matrix multiplication
    *4. The result in [(nData*nHeight*nWidth),nOuput_channel], reshape it
    *************************************/
    cout<<kernel_size<<endl;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int k=0;k<height_stride_len;k+=this->stride){
            for(unsigned int p=0;p<width_stride_len;p+=this->stride){
                for(unsigned int l=0;l<this->input_channel;l++){
                    if((i==0)&&(k==0)&&(p==0)){
                        for(unsigned int h=0;h<output_channel;h++){
                            this_convolution_weight = this->conv_weights[l][h];
                            this_convolution_weight_streched = mat_reshape(this_convolution_weight,this->kernel_size*this->kernel_size,1,1);
                            reshaped_weights_channels[l].col(h) = this_convolution_weight_streched;
                        }
                    }
                    //The weight of the lth input channel
                    current_input_channel = padded_x[i][l];
                    convolution_window_data = mat_extract(current_input_channel,k+1,k+kernel_size,p+1,p+kernel_size);
                    this_convolution_streched = mat_reshape(convolution_window_data,1,this->kernel_size*this->kernel_size,1);
                    int row_ind = i*nData + k*height_stride_len + p;  //axis = 1
                    reshaped_padded_input_channels[l].row(row_ind) = this_convolution_streched;
                }
            }
        }
    }
    //Define the final reshaped input data and  reshaped weights
    MatrixXmat reshaped_padded_input;
    MatrixXmat reshaped_weights;
    //concatenate all the data
    for(unsigned int i=0;i<this->input_channel;i++){
        if(i==0){
            reshaped_padded_input = reshaped_padded_input_channels[i];
            reshaped_weights = reshaped_weights_channels[i];
        }
        else{
            reshaped_padded_input = mat_concate(reshaped_padded_input,reshaped_padded_input_channels[i],1);
            reshaped_weights = mat_concate(reshaped_weights,reshaped_weights_channels[i],0);
        }
    }
    //perform multiplication
    MatrixXmat rst_stacked = reshaped_padded_input*reshaped_weights;
    MatrixXmat this_image_channels;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int k=0;k<height_stride_len;k+=this->stride){
            for(unsigned int p=0;p<width_stride_len;p+=this->stride){
                int recover_ind = i*nData + k*height_stride_len + p;
                this_image_channels = rst_stacked.row(recover_ind);
                for(unsigned int j=0;j<this->output_channel;j++){
                    convolution_data[i][j](k,p) = this_image_channels(0,j);
                }
            }
        }
    }

    return convolution_data;
}

vector<vector<MatrixXmat> > Convolution_layer::backward_func(vector<vector<MatrixXmat> > dLdy){
    //The input dLdy show be [nData * output_channel * new_height * new_width]
    //Input image is [nData * input_channel * height * width]
    int nData = this->x.size();
    //size of the convolutioned layer
    int height_new = dLdy[0][0].rows();
    int width_new = dLdy[0][0].cols();
    //size of the previous image
    int height = this->x[0][0].rows();
    int width = this->x[0][0].cols();
    //size of the padded iamge
    int height_padded = this->padded_x[0][0].rows();
    int width_padded = this->padded_x[0][0].cols();
    //Construct the temperaty 4-d matrix to reduce the usage of memory
    vector<vector<MatrixXmat> > temp_dLdw(this->input_channel,vector<MatrixXmat>(this->output_channel,\
                                                            MatrixXmat::Zero(this->kernel_size,this->kernel_size)));
    vector<vector<MatrixXmat> > dLdx(nData,vector<MatrixXmat>(this->input_channel,\
                                                        MatrixXmat::Zero(height,width)));
    MatrixXmat this_convulution_field;
    MatrixXmat this_dLdw;
    /******************This is a bittle bit complex*****************
    *This->conv_gradient:
            This should be [input_channel * output_channel * kernel_size * kernel_size]. We notice that for each output channel
            we could compute the gradient seperately
    *This->dLdx:
            This should be in the size of [nData * input_channel * height * width]. We should compute this for each input_channel
            separately, adding up the gradients of coming from the output channels
    ********************************************************/
    for(unsigned int i=0;i<nData;i++){
        //for each input channel
        for(unsigned int j=0;j<this->input_channel;j++){
            //for each output channel
            for(unsigned int k=0;k<this->output_channel;k++){
                MatrixXmat this_dLdx = MatrixXmat::Zero(height_padded,width_padded);
                //Looping the new image
                for(unsigned int l_h=0;l_h<height_new;l_h++){
                    for(unsigned int l_w=0;l_w<width_new;l_w++){
                        //Update dLdw
                        //get the current start/end row/column for the correspoding oringinal matrix
                        int start_row = l_h*(this->stride)+1;
                        int end_row = start_row +(this->kernel_size)-1;
                        int start_column = l_w*(this->stride)+1;
                        int end_column = start_column + (this->kernel_size)-1;
                        this_convulution_field = mat_extract(this->padded_x[i][j],start_row,\
                                                             end_row,start_column,end_column);
                        this_dLdw = dLdy[i][k](l_h,l_w)*this_convulution_field;
                        temp_dLdw[j][k] = temp_dLdw[j][k] + this_dLdw;
                        //Update dLdx
                        for(unsigned int p_h=0;p_h<this->kernel_size;p_h++){
                            for(unsigned int p_l=0;p_l<this->kernel_size;p_l++){
                                int back_prop_row = p_h+start_row-1;
                                int back_prop_column = p_l+start_column-1;
                                this_dLdx(back_prop_row,back_prop_column) += dLdy[i][k](l_h,l_w)*this->conv_weights[j][k](p_h,p_l);
                            }
                        }
                    }
                }
                //take out the padding gradient (only maintain gradient for the plain input) and give the value to dLdx
                dLdx[i][j] = dLdx[i][j] + mat_extract(this_dLdx,this->padding_height+1,height_padded-this->padding_height,\
                                             this->padding_width+1,width_padded-this->padding_width);
            }
        }
    }
    this->conv_gradient = temp_dLdw;

    return dLdx;
}


/**************base-pooling Layer Codes****************/
pooling_layer::pooling_layer(int kernel_size, int stride, string overlap_mode, string padding_mode){
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->overlap_mode = overlap_mode;
    this->padding_mode = padding_mode;
}

pooling_layer::~pooling_layer(){

}

vector<vector<MatrixXmat> > pooling_layer::forward_func(vector<vector<MatrixXmat> > x){
    //input should be [nData * nChannel * height * width]
    //get the input shape information
    /**************Logic of 'overlap mode' and 'padding mode'*************
    *If overlap mode == separate:
    *       No sence for the 'padding mode' keyword, here we only padding if the image is not suitable for pooling
    *If overlap mode == overlap:
    *       The 'padding mode' keyword acting as the same function in convolutional layer
    **********************************************************************/
    int nData = x.size();
    int nChannel = x[0].size();
    int height = x[0][0].rows();
    int width = x[0][0].cols();
    int stride = 0;
    this->x = x;
    //padded data construction
    vector<vector<MatrixXmat> > padded_x;
    //Padding rows and columns
    int padding_height = 0;
    int padding_width = 0;
    //judging the mode
    if(this->overlap_mode=="overlap"){
        stride = this->stride;
        if(this->padding_mode == "same"){
            padding_width = (this->kernel_size + (stride-1)*width - stride)/2;
            padding_height = (this->kernel_size + (stride-1)*height - stride)/2;
            padded_x = data_padding(x,padding_height,padding_width);
        }
        else if(this->padding_mode == "valid"){
            padded_x = x;
        }
        else{
            throw std::invalid_argument("padding mode unrecognised!");
        }
    }
    else if(this->overlap_mode=="separate"){
        stride = this->kernel_size;
        while((height+2*padding_height)%this->kernel_size!=0){
            padding_height += 1;
        }
        while((width+2*padding_width)%this->kernel_size!=0){
            padding_width += 1;
        }
        padded_x = data_padding(x,padding_height,padding_width);
    }
    else{
        throw std::invalid_argument("overlap mode unrecognised!");
    }
    //update the variables for the layer
    this->padding_height = padding_height;
    this->padding_width = padding_width;
    this->stride = stride; //store the value of stride to perform back_propagation
    this->padded_x = padded_x;
    //shape of the new output
    int new_height = (int)((height+2*padding_height-(this->kernel_size))/stride) + 1;
    int new_width = (int)((width+2*padding_width-(this->kernel_size))/stride) + 1;
    //construct the optput data in shape
    vector<vector<MatrixXmat> > pooled_output(nData,vector<MatrixXmat>(nChannel,\
                                                MatrixXmat::Zero(new_height,new_width)));
    //construct the temporary dydx
    vector<vector<vector<vector<MatrixXmat> > > > temp_dydx(nData,vector<vector<vector<MatrixXmat> > >\
                                                       (nChannel,vector<vector<MatrixXmat> >(new_height,\
                                                        vector<MatrixXmat>(new_width,\
                                                        MatrixXmat::Zero(this->kernel_size,this->kernel_size)))));
    //the temperary image
    MatrixXmat this_image;
    //struct to store the pooling information
    pooling_information pooling_this_image;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nChannel;j++){
            for(unsigned int k=0;k<new_height;k+=stride){
                for(unsigned int p=0;p<new_width;p+=stride){
                    int start_row = stride*k + 1;
                    int end_row = start_row+(this->kernel_size)-1;
                    int start_column = stride*p+1;
                    int end_column = start_column+(this->kernel_size)-1;
                    this_image = mat_extract(padded_x[i][j],start_row,end_row,start_column,end_column);
                    pooling_this_image = this->get_pooling(this_image);
                    pooled_output[i][j](k,p) = pooling_this_image.pooling_value;
                    temp_dydx[i][j][k][p] = pooling_this_image.this_dydx;
                }
            }
        }
    }
    this->dydx = temp_dydx;

    return pooled_output;
}

//backward routine
vector<vector<MatrixXmat> > pooling_layer::backward_func(vector<vector<MatrixXmat> > dLdy){
    //input should be [nData * nChannel * new_height * new_width]
    //output should be [nData * nChannel * new_height * new_width]
    //get the input data amount and channels
    int nData = dLdy.size();
    int nChannel = dLdy[0].size();
    //get the height and width information
    int height = this->x[0][0].rows();
    int width = this->x[0][0].cols();
    //the shape of input pooled data
    int new_height = dLdy[0][0].rows();
    int new_width = dLdy[0][0].cols();
    //the shape of padded data
    int padded_height = this->padded_x[0][0].rows();
    int padded_width = this->padded_x[0][0].cols();
    vector<vector<MatrixXmat> > dLdx(nData,vector<MatrixXmat>(nChannel,\
                                    MatrixXmat::Zero(height,width)));
    vector<vector<MatrixXmat> > dLdx_padded(nData,vector<MatrixXmat>(nChannel,\
                                    MatrixXmat::Zero(padded_height,padded_width)));
    //contrianer that recover the current gradient
    MatrixXmat this_dydx;
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nChannel;j++){
            for(unsigned int k=0;k<new_height;k++){
                for(unsigned int p=0;p<new_width;p++){
                    //calculate the corresponding area in the original image
                    this_dydx = this->dydx[i][j][k][p];
                    int start_row = k*(this->stride);
                    int end_row = start_row + (this->kernel_size);
                    int start_column = p*(this->stride);
                    int end_column = start_column + (this->kernel_size);
                    for(unsigned int l_h=start_row;l_h<end_row;l_h++){
                        for(unsigned int l_w=start_column;l_w<end_column;l_w++){
                            int dydx_row = l_h - start_row;
                            int dydx_column = l_w - start_column;
                            dLdx_padded[i][j](l_h,l_w) += this_dydx(dydx_row,dydx_column)*dLdy[i][j](k,p);
                        }
                    }
                }
            }
            dLdx[i][j] = mat_extract(dLdx_padded[i][j],this->padding_height+1,padded_height-this->padding_height,\
                                         this->padding_width+1,padded_width-this->padding_width);
        }
    }



    return dLdx;
}

pooling_information pooling_layer::get_pooling(MatrixXmat current_image){
    //Input is a 2-d extracted area of the image
    //This is the basic pooling layer which only return the first value
    //find the shape
    int height = current_image.rows();
    int width = current_image.cols();
    //define the structure to store the informations
    pooling_information this_pooling_information;
    //get the pooled value
    this_pooling_information.pooling_value = current_image(0,0);
    //get the dydx backprop matrix
    MatrixXmat this_dydx = MatrixXmat::Zero(height,width);
    this_dydx(0,0)=1;
    this_pooling_information.this_dydx = this_dydx;

    return this_pooling_information;
}


/*****************Codes for Max-pooling*****************/
max_pooling_layer::~max_pooling_layer(){

}

pooling_information max_pooling_layer::get_pooling(MatrixXmat current_image){
    //find the shape
    int height = current_image.rows();
    int width = current_image.cols();
    //define the structure to store the informations
    pooling_information this_pooling_information;
    //define initial guessing of maximum
    //find the maximum value
    MatrixXmat::Index maxRow, maxCol;
    UnitType max_value = current_image.maxCoeff(&maxRow, &maxCol);
    int max_row_ind = maxRow;
    int max_column_ind = maxCol;
    MatrixXmat dydx_mat = MatrixXmat::Zero(height,width);
    dydx_mat(max_row_ind,max_column_ind) = 1;
    //assign the value to the structure
    this_pooling_information.pooling_value = max_value;
    this_pooling_information.this_dydx = dydx_mat;

    return this_pooling_information;
}

/**************codes for sum-pooling***************/
sum_pooling_layer::~sum_pooling_layer(){

}

pooling_information sum_pooling_layer::get_pooling(MatrixXmat current_image){
    //find the shape
    int height = current_image.rows();
    int width = current_image.cols();
    //define the structure to store the informations
    pooling_information this_pooling_information;
    //variable to store the summation value
    UnitType pooling_value = current_image.sum();
    this_pooling_information.pooling_value = pooling_value;
    this_pooling_information.this_dydx = MatrixXmat::Ones(height,width);

    return this_pooling_information;
}

/************codes for average-pooling layer*************/
average_pooling_layer::~average_pooling_layer(){

}

pooling_information average_pooling_layer::get_pooling(MatrixXmat current_image){
    //find the shape
    int height = current_image.rows();
    int width = current_image.cols();
    UnitType total_elements = (UnitType)(height*width);
    //define the structure to store the informations
    pooling_information this_pooling_information;
    //variable to store the summation value
    UnitType pooling_value = current_image.mean();
    this_pooling_information.pooling_value = pooling_value;
    this_pooling_information.this_dydx = (1/total_elements)*MatrixXmat::Ones(height,width);

    return this_pooling_information;
}

/**********Codes for Stochastical Pooling********/
//destructor
stochastic_pooling_layer::~stochastic_pooling_layer(){
}

pooling_information stochastic_pooling_layer::get_pooling(MatrixXmat current_image){
    //find the shape
    int height = current_image.rows();
    int width = current_image.cols();
    MatrixXmat prob_mat = MatrixXmat::Zero(height,width);
    //define the structure to store the informations
    pooling_information this_pooling_information;
    //define initial guessing of maximum
    //sum all activations to compute probability
    UnitType sum_activation = current_image.sum();
    //compute the probability and assign the sector
    UnitType culmul_value = 0;
    MatrixXmat culmul_prob = MatrixXmat::Zero(height,width);
    for(unsigned int i=0;i<height;i++){
        for(unsigned int j=0;j<width;j++){
            prob_mat(i,j) = current_image(i,j)/sum_activation;
            culmul_value += prob_mat(i,j);
            culmul_prob(i,j) = culmul_value;
        }
    }
    //sample the index
    int row_ind;
    int column_ind;
    UnitType rand_num = ((UnitType) rand() / (RAND_MAX));
    //define pooling value
    UnitType pool_value = 0;
    //Sampling
    for(unsigned int i=0;i<height;i++){
        for(unsigned int j=0;j<width;j++){
            if(rand_num<=culmul_prob(i,j)){
                row_ind = i;
                column_ind = j;
                pool_value = current_image(i,j);
            }
            else{
            }
        }
    }
    MatrixXmat dydx_mat = MatrixXmat::Zero(height,width);
    dydx_mat(row_ind,column_ind) = 1;
    //assign the value to the structure
    this_pooling_information.pooling_value = pool_value;
    this_pooling_information.this_dydx = dydx_mat;

    return this_pooling_information;
}
