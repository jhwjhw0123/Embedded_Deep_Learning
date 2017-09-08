/*******This demo shows a three-hidden layer neural network********
layer1: linear 1000 nodes + RELU
layer2: linear 500 nodes + RELU
layer3: linear 100 nodes + RELU
output_layer: softmax cross-entropy layer
*****************************************************************/

#include <cmath>
#include <iostream>
#include <ctime>
#include <vector>
#include <cstdlib>
#define PI 3.1415927
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "DNN.h"
#include "CNN.h"
#include "RNN.h"
#include "Optimizer.h"
#include "MatrixOperation.h"
#include "MathType.h"
#include "mnist/mnist_reader.hpp"
// #include "jsonLib/json.hpp"
#include "ParaSave.h"


using namespace std;
using namespace Eigen;
using namespace DNN;
using namespace CNN;
using namespace RNN;

template<typename decimal>
MatrixXmat pre_process_images(vector<vector<unsigned char> > data_set, int nImages){
    unsigned int nData = data_set.size();
    unsigned int nDim = data_set[0].size();
    MatrixXmat temp_data_mat(nData,nDim);
    for(unsigned int i=0;i<nData;i++){
        for(unsigned int j=0;j<nDim;j++){
            temp_data_mat(i,j) = (decimal)data_set[i][j];
            }
    }
    MatrixXmat selected_data_set = mat_extract(temp_data_mat,1,nImages,1,nDim);

    return selected_data_set;
}

MatrixXmat pre_process_labels(vector<unsigned char> label_set, int nImages, int nClasses){
    unsigned int nData = label_set.size();
    MatrixXmat temp_label_mat = MatrixXmat::Zero(nData,nClasses);
    for(unsigned int i=0;i<nData;i++){
        int this_label = (int)(label_set[i]);
        temp_label_mat(i,this_label) = 1;
    }

    MatrixXmat selected_label_set = mat_extract(temp_label_mat,1,nImages,1,nClasses);

    return selected_label_set;
}

vector<int> stochastic_index(int nData, int batch_size){
    vector<int> index_vec(batch_size);
    for(unsigned int i=0;i<batch_size;i++){
        index_vec[i] = (int)(rand()%nData);
    }

    return index_vec;
}

MatrixXmat stochastic_batch_data_sampling(MatrixXmat data_set, vector<int> stochastic_index){
    unsigned int batch_size = stochastic_index.size();
    unsigned int nDim = data_set.cols();
    MatrixXmat batch_data(batch_size,nDim);
    for(unsigned int i=0;i<batch_size;i++){
        int this_index = stochastic_index[i];
        batch_data.row(i) = data_set.row(this_index);
    }

    return batch_data;
}

MatrixXmat stochastic_batch_label_sampling(MatrixXmat label_set, vector<int> stochastic_index){
    unsigned int batch_size = stochastic_index.size();
    unsigned int nClasses = label_set.cols();
    MatrixXmat batch_label(batch_size,nClasses);
    for(unsigned int i=0;i<batch_size;i++){
        int this_index = stochastic_index[i];
        batch_label.row(i) = label_set.row(this_index);
    }

    return batch_label;
}

MatrixXmat one_hot_prediction_encoding(MatrixXi prediction, int n_classes){
    //Input vector is [n_data * 1], encode it to [n_data * n_classes] one_hot representation
    unsigned int nData = prediction.rows();
    MatrixXmat encoded_prediction = MatrixXmat::Zero(nData,n_classes);
    for(unsigned int i=0;i<nData;i++){
            int ind = prediction(i,0);
            encoded_prediction(i,ind) = 1;
    }

    return encoded_prediction;
}

void fun_dnn(){
    //Set how many data to be used, change here is you want
    int train_amount = 10000;
    int test_amount = 1000;
    //read the data
    cout<<"Reading the Data..."<<endl;
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test = pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    cout<<"Data Reading Finished!"<<endl;
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    unsigned int max_interation = 100;
    int batch_size = 256;
    UnitType learning_rate_1 = 1e-3;
    UnitType learning_rate_2 = 1e-3;
    UnitType learning_rate_3 = 1e-4;
    UnitType learning_rate_4 = 1e-5;
    //Consruct the layers
    Linear_Layer layer_1_trans(nDim,1000);
    RELU_Layer layer_1_activ(1000);
    Linear_Layer layer_2_trans(1000,500);
    RELU_Layer layer_2_activ(500);
    Linear_Layer layer_3_trans(500,100);
    RELU_Layer layer_3_activ(100);
    Linear_Layer output_trans_layer(100,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //construct optimizers
    Adam_optimizer<UnitType> opt_layer_1(learning_rate_1,1,0.9,0.999,1e-6);//,0.9,0.999,1e-6);
    Adam_optimizer<UnitType> opt_layer_2(learning_rate_2,1,0.9,0.999,1e-6);//,0.999,1e-6);
    Adam_optimizer<UnitType> opt_layer_3(learning_rate_3,1,0.9,0.999,1e-6);
//    Adadelta_optimizer<UnitType> opt_layer_1(learning_rate,1,1e-6,0.9);
//    Adadelta_optimizer<UnitType> opt_layer_2(learning_rate,1,1e-6,0.9);
    //Adadelta_optimizer<UnitType> opt_layer_3(learning_rate,1,1e-6,0.9);
    //Adadelta_optimizer<UnitType> opt_trans_layer(learning_rate,1,1e-6,0.9);
    //sgd_optimizer<UnitType> opt_layer_1(learning_rate,1);
    //sgd_optimizer<UnitType> opt_layer_2(learning_rate,1);
//    sgd_optimizer<UnitType> opt_trans_layer(learning_rate,1);
    Adam_optimizer<UnitType> opt_trans_layer(learning_rate_4,1,0.9,0.999,1e-6);//,0.9,0.999,1e-6);
    //define the containers to process the data
    //container of input
    vector<int> index_this_batch(batch_size);
    MatrixXmat data_this_batch(batch_size,nDim);
    MatrixXmat label_this_batch(batch_size,nClasses);
    /*********Foward function containers*********/
    //containers of the first layer
    MatrixXmat layer_1_input;
    MatrixXmat layer_1_output;
    //containers of the second layer
    MatrixXmat layer_2_input;
    MatrixXmat layer_2_output;
    //containers of the third layer
    MatrixXmat layer_3_input;
    MatrixXmat layer_3_output;
    //containers of the output layer
    MatrixXmat layer_out_input(batch_size,nClasses);
    MatrixXi prediction_output;
    /************backward function containers***************/
    //containers for output layer
    MatrixXmat back_prop_out_layer;
    MatrixXmat back_prop_out_trans;
    //containers for layer 3
    MatrixXmat back_prop_layer_3_act;
    MatrixXmat back_prop_layer_3_trans;
    //containers for layer 2
    MatrixXmat back_prop_layer_2_act;
    MatrixXmat back_prop_layer_2_trans;
    //containers for layer 1
    MatrixXmat back_prop_layer_1_act;
    MatrixXmat back_prop_layer_1_trans;
    //Variables to compute accuracy
    MatrixXmat one_hot_prediction_train;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    clock_t t_batch_start,t_batch_end;
    //inference and train
    for(unsigned int i=0;i<max_interation;i++){
        //perform stochastic gradient descent
        cout<<"Carrying out the "<<i+1<<" iteration"<<endl;
        prev_test_accuracy = train_accuracy;
        unsigned int batch_numbers = train_amount/batch_size;
        UnitType itr_loss = 0;
        for(unsigned int j=0;j<batch_numbers;j++){
            srand( (unsigned)time( NULL ));
            if(j==0){
	    				t_batch_start=clock();
	    			}
            index_this_batch = stochastic_index(train_amount,batch_size);
            data_this_batch = stochastic_batch_data_sampling(data_train,index_this_batch);
            label_this_batch = stochastic_batch_label_sampling(label_train,index_this_batch);
            /*****************Forward routine***************/
            //layer 1
            layer_1_input = layer_1_trans.forward_func(data_this_batch);
            layer_1_output = layer_1_activ.forward_func(layer_1_input);
            //layer 2
            layer_2_input = layer_2_trans.forward_func(layer_1_output);
            layer_2_output = layer_2_activ.forward_func(layer_2_input);
            //layer 3
            layer_3_input = layer_3_trans.forward_func(layer_2_output);
            layer_3_output = layer_3_activ.forward_func(layer_3_input);
            //output layer
            layer_out_input = output_trans_layer.forward_func(layer_3_output);
            //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
            prediction_output = output_layer.output_func(layer_out_input,label_this_batch);
            itr_loss += output_layer.loss;
            /************Backward Routine*************/
            //loss layer
            back_prop_out_layer = output_layer.backward_func();
            back_prop_out_trans = output_trans_layer.backward_func(back_prop_out_layer);
            //train
            output_trans_layer.train_func(opt_trans_layer,"None",0.01);
            //layer 3
            back_prop_layer_3_act = layer_3_activ.backward_func(back_prop_out_trans);
            back_prop_layer_3_trans = layer_3_trans.backward_func(back_prop_layer_3_act);
            //train
            layer_3_trans.train_func(opt_layer_3,"None",0.01);
            //cout<<"9"<<endl;
            //layer 2
            back_prop_layer_2_act = layer_2_activ.backward_func(back_prop_layer_3_trans);
            back_prop_layer_2_trans = layer_2_trans.backward_func(back_prop_layer_2_act);
            //cout<<"10"<<endl;
            //train
            layer_2_trans.train_func(opt_layer_2,"None",0.01);
            //layer 1
            back_prop_layer_1_act = layer_1_activ.backward_func(back_prop_layer_2_trans);
            back_prop_layer_1_trans = layer_1_trans.backward_func(back_prop_layer_1_act);
            //train
            layer_1_trans.train_func(opt_layer_1,"None",0.01);
            cout<<output_layer.loss<<endl;
            if(j==0){
	    			 	t_batch_end=clock();
					    float diff_train_batch = ((float)t_batch_end-(float)t_batch_start);
					    float seconds_train_batch = diff_train_batch / CLOCKS_PER_SEC;
					    cout<<"Single Batch Running Time is:"<<seconds_train_batch<<endl;
    				}
        }
        //print loss
        cout<<"The trainning of the "<<i+1<<" iteration has finished!"<<endl;
        cout<<"The loss of the "<<i+1<<" iteration is: "<<(double)(itr_loss)<<endl;
        /******Calculate the training accuracy for all the data*******/
        //layer 1
        clock_t t_1,t_2,t_3,t_4,t_5;
        t_1 = clock();
        layer_1_input = layer_1_trans.forward_func(data_train);
        t_2 = clock();
        float diff_linear_layer = ((float)t_2-(float)t_1);
		    float seconds_linear = diff_linear_layer / CLOCKS_PER_SEC;
		    cout<<"The Linear Layer Inference time is:"<<seconds_linear<<endl;
        layer_1_output = layer_1_activ.forward_func(layer_1_input);
        //layer 2
        layer_2_input = layer_2_trans.forward_func(layer_1_output);
        layer_2_output = layer_2_activ.forward_func(layer_2_input);
        //layer 3
        layer_3_input = layer_3_trans.forward_func(layer_2_output);
        layer_3_output = layer_3_activ.forward_func(layer_3_input);
        //output layer
        layer_out_input = output_trans_layer.forward_func(layer_3_output);
        //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
        prediction_output = output_layer.output_func(layer_out_input,label_train);
        one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
        train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
        //print out the information
        cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
        t_3 = clock();
        float diff_train = ((float)t_3-(float)t_1);
		    float seconds_train = diff_train / CLOCKS_PER_SEC;
		    cout<<"The Train Data Inference time is:"<<seconds_train<<endl;
        /******Calculate the training accuracy for all the data*******/
        //layer 1
        t_4 = clock();
        layer_1_input = layer_1_trans.forward_func(data_test);
        layer_1_output = layer_1_activ.forward_func(layer_1_input);
        //layer 2
        layer_2_input = layer_2_trans.forward_func(layer_1_output);
        layer_2_output = layer_2_activ.forward_func(layer_2_input);
        //layer 3
        layer_3_input = layer_3_trans.forward_func(layer_2_output);
        layer_3_output = layer_3_activ.forward_func(layer_3_input);
        //output layer
        layer_out_input = output_trans_layer.forward_func(layer_3_output);
        //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
        prediction_output = output_layer.output_func(layer_out_input,label_train);
        one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
        test_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_test);
        cout<<"The Test accuracy of the "<<i+1<<" iteration is: "<<(double)(test_accuracy)<<endl;
        t_5 = clock();
        float diff_test = ((float)t_5-(float)t_4);
		    float seconds_test = diff_test / CLOCKS_PER_SEC;
		    cout<<"The Train Data Inference time is:"<<seconds_test<<endl;
        //If sufficiently high test accuracy and start to overfit
        if(test_accuracy>=0.95&&test_accuracy<=prev_test_accuracy){
            vector<MatrixXmat> para_container;
            Save_mat(layer_1_trans.Para_copy_store(),"paramters_text/DNN/W_1.txt");
            Save_mat(layer_2_trans.Para_copy_store(),"paramters_text/DNN/W_2.txt");
            Save_mat(layer_3_trans.Para_copy_store(),"paramters_text/DNN/W_3.txt");
            Save_mat(output_trans_layer.Para_copy_store(),"paramters_text/DNN/W_4.txt");
            break;
        }
    }
    cout<<"DNN Finished and parameter saved! Press any key + enter to finish the program."<<endl;
}

//Test a demo of CNN model
void fun_cnn(){
/****************
*CNN convolutional layer construction instruction
*input_channels, output_channels, kernel_size, stride, mode(string)
*************/
//construct the CNN
//Set how many data to be used, change here is you want
int train_amount = 10000;
int test_amount = 1000;
//read the data
auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
//train data in [n_trainData * nDim] shape
MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
//test data in [n_testData * nDim] shape
MatrixXmat data_test = pre_process_images<UnitType>(dataset.test_images,test_amount);
//train labels in [n_trainData * 1] shape
MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
//test labels in [n_testData * 1] shape
MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
//Get hyper-parameters
int nDim = data_train.cols();
int nClasses = 10;
unsigned int max_interation = 100;
int batch_size = 256;
UnitType learning_rate = 1e-4;
//Construct Layers
Convolution_layer conv_layer_1(1,3,3,1,"same");
max_pooling_layer pool_layer_1(2,1,"separate","same");
Convolution_layer conv_layer_2(3,9,3,1,"same");
max_pooling_layer pool_layer_2(2,1,"separate","same");
Linear_Layer fully_connected_layer(7*7*9,nClasses);
soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
//Optimizer
Adam_optimizer<UnitType> opt_trans_layer(learning_rate,1,0.9,0.999,1e-8);
Adam_optimizer<UnitType> opt_convolutional_layer_1(learning_rate,3,0.9,0.999,1e-8);
Adam_optimizer<UnitType> opt_convolutional_layer_2(learning_rate,27,0.9,0.999,1e-8);
//define the containers to process the data
//container of input
vector<int> index_this_batch(batch_size);
MatrixXmat data_this_batch(batch_size,nDim);
MatrixXmat label_this_batch(batch_size,nClasses);
vector<vector<MatrixXmat> > convolution_input;
vector<vector<MatrixXmat> > convolution_output;
vector<vector<MatrixXmat> > pooling_output;
MatrixXmat linear_input;
MatrixXmat linear_output;
MatrixXi prediction_output;
//containers for the backward routine
MatrixXmat back_prop_out_layer;
MatrixXmat back_prop_linear_layer;
vector<vector<MatrixXmat> > back_prop_conv_flat;
vector<vector<MatrixXmat> > back_prop_pooling_2;
vector<vector<MatrixXmat> > back_prop_conv_layer_2;
vector<vector<MatrixXmat> > back_prop_pooling_1;
vector<vector<MatrixXmat> > back_prop_conv_layer_1;  //only for containing
//test the accuracy
MatrixXmat one_hot_prediction_train;
MatrixXmat one_hot_prediction_test;
UnitType train_accuracy;
UnitType test_accuracy;
UnitType prev_test_accuracy;
clock_t t_batch_start,t_batch_end;
//inference and train
for(unsigned int i=0;i<max_interation;i++){
    //perform stochastic gradient descent
    cout<<"Carrying out the "<<i+1<<" iteration"<<endl;
    prev_test_accuracy = test_accuracy;
    unsigned int batch_numbers = train_amount/batch_size;
    UnitType itr_loss = 0;
    for(unsigned int j=0;j<batch_numbers;j++){
        srand( (unsigned)time( NULL ));
        if(j==0){
	    			t_batch_start=clock();
	    	}
        index_this_batch = stochastic_index(train_amount,batch_size);
        data_this_batch = stochastic_batch_data_sampling(data_train,index_this_batch);
        label_this_batch = stochastic_batch_label_sampling(label_train,index_this_batch);
        /*********Tranform data**********/
        convolution_input = data_squaring(data_this_batch,28,28,1);
        /*****************Forward routine***************/
        //cout<<"1"<<endl;
        //Conv 1
        convolution_output = conv_layer_1.forward_func(convolution_input);
        //cout<<"2"<<endl;
        //pooling
        pooling_output = pool_layer_1.forward_func(convolution_output);
        //cout<<"3"<<endl;
         //Conv 2
        convolution_output = conv_layer_2.forward_func(pooling_output);
        //cout<<"4"<<endl;
        //pooling
        pooling_output = pool_layer_2.forward_func(convolution_output);
        //cout<<"5"<<endl;
        //flatten the data
        linear_input = data_flatting(pooling_output);
        //cout<<"6"<<endl;
        linear_output = fully_connected_layer.forward_func(linear_input);
        //cout<<"7"<<endl;
        //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
        prediction_output = output_layer.output_func(linear_output,label_this_batch);
        itr_loss += output_layer.loss;
        /************Backward Routine*************/
        //loss layer
        back_prop_out_layer = output_layer.backward_func();
        back_prop_linear_layer = fully_connected_layer.backward_func(back_prop_out_layer);
        //cout<<"8"<<endl;
        //train
        fully_connected_layer.train_func(opt_trans_layer,"None",0.01);
        //transfer the gradient
        //[nData * 9 * 7 * 7]
        back_prop_conv_flat = data_squaring(back_prop_linear_layer,7,7,9);
        //cout<<"9"<<endl;
        back_prop_pooling_2 = pool_layer_2.backward_func(back_prop_conv_flat);
        back_prop_conv_layer_2 = conv_layer_2.backward_func(back_prop_pooling_2);
        //cout<<"10"<<endl;
        back_prop_pooling_1  = pool_layer_1.backward_func(back_prop_conv_layer_2);
        back_prop_conv_layer_1 = conv_layer_1.backward_func(back_prop_pooling_1);
        //cout<<"11"<<endl;
        conv_layer_1.train(opt_convolutional_layer_1,"None",0.01);
        conv_layer_2.train(opt_convolutional_layer_2,"None",0.01);
        cout<<(double)(output_layer.loss)<<endl;
        if(j==0){
	    			 	t_batch_end=clock();
					    float diff_train_batch = ((float)t_batch_end-(float)t_batch_start);
					    float seconds_train_batch = diff_train_batch / CLOCKS_PER_SEC;
					    cout<<"Single Batch Running Time is:"<<seconds_train_batch<<endl;
    				}
        }
    //print loss
    cout<<"The trainning of the "<<i+1<<" iteration has finished!"<<endl;
    cout<<"The loss of the "<<i+1<<" iteration is: "<<(double)(itr_loss)<<endl;
    clock_t t1,t2;
    t1=clock();
    /******Calculate the training accuracy for all the data*******/
    convolution_input = data_squaring(data_train,28,28,1);
    //layer 1
    convolution_output = conv_layer_1.forward_func(convolution_input);
    //pooling
    pooling_output = pool_layer_1.forward_func(convolution_output);
    //layer2
    convolution_output = conv_layer_2.forward_func(pooling_output);
    //pooling
    pooling_output = pool_layer_2.forward_func(convolution_output);
    //flatten the data
    linear_input = data_flatting(pooling_output);
    linear_output = fully_connected_layer.forward_func(linear_input);
    //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
    prediction_output = output_layer.output_func(linear_output,label_train);
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
    t2=clock();
    float diff_train = ((float)t2-(float)t1);
    float seconds_train = diff_train / CLOCKS_PER_SEC;
    cout<<"The training data Inference time is:"<<seconds_train<<endl;
    /**************calculste the test accuracy****************/
    clock_t t3,t4;
    t3=clock();
    convolution_input = data_squaring(data_test,28,28,1);
    //layer 1
    convolution_output = conv_layer_1.forward_func(convolution_input);
    //pooling
    pooling_output = pool_layer_1.forward_func(convolution_output);
    //layer2
    convolution_output = conv_layer_2.forward_func(pooling_output);
    //pooling
    pooling_output = pool_layer_2.forward_func(convolution_output);
    //flatten the data
    linear_input = data_flatting(pooling_output);
    linear_output = fully_connected_layer.forward_func(linear_input);
    //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
    prediction_output = output_layer.output_func(linear_output,label_test);
    one_hot_prediction_test = one_hot_prediction_encoding(prediction_output,nClasses);
    test_accuracy = output_layer.acc_classfication(one_hot_prediction_test,label_test);
    t4=clock();
    float diff_test = ((float)t4-(float)t3);
    float seconds_test = diff_test / CLOCKS_PER_SEC;
    cout<<"The training data Inference time is:"<<seconds_test<<endl;
    if(test_accuracy>=0.945&&test_accuracy<=prev_test_accuracy){
        string file_path = "paramters_text/CNN/";
        Save_mat(conv_layer_1.Para_copy_store(),file_path+"W_conv_1.txt");
        Save_mat(conv_layer_2.Para_copy_store(),file_path+"W_conv_2.txt");
        Save_mat(fully_connected_layer.Para_copy_store(),file_path+"W_1.txt");
        break;
        }
    //print out the information
    cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
    cout<<"The Test accuracy of the "<<i+1<<" iteration is: "<<(double)(test_accuracy)<<endl;
    }
    cout<<"CNN program finished and the parameters have been saved!"<<endl;
}


//Test a demo of CNN model (Without Pooling)
void fun_cnn_without_pool(){
/****************
*CNN convolutional layer construction instruction
*input_channels, output_channels, kernel_size, stride, mode(string)
*************/
//construct the CNN
//Set how many data to be used, change here is you want
int train_amount = 10000;
int test_amount = 1000;
//read the data
auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
//train data in [n_trainData * nDim] shape
MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
//test data in [n_testData * nDim] shape
MatrixXmat data_test = pre_process_images<UnitType>(dataset.test_images,test_amount);
//train labels in [n_trainData * 1] shape
MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
//test labels in [n_testData * 1] shape
MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
//Get hyper-parameters
int nDim = data_train.cols();
int nClasses = 10;
unsigned int max_interation = 100;
int batch_size = 256;
UnitType learning_rate = 1e-3;
//Construct Layers
Convolution_layer conv_layer_1(1,3,3,1,"same");
max_pooling_layer pool_layer_1(2,1,"separate","same");
Convolution_layer conv_layer_2(3,9,3,1,"same");
max_pooling_layer pool_layer_2(2,1,"separate","same");
Linear_Layer fully_connected_layer(28*28*9,nClasses);
soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
//Optimizer
Adam_optimizer<UnitType> opt_trans_layer(learning_rate,1,0.9,0.999,1e-8);
Adam_optimizer<UnitType> opt_convolutional_layer_1(learning_rate,3,0.9,0.999,1e-8);
Adam_optimizer<UnitType> opt_convolutional_layer_2(learning_rate,27,0.9,0.999,1e-8);
//define the containers to process the data
//container of input
vector<int> index_this_batch(batch_size);
MatrixXmat data_this_batch(batch_size,nDim);
MatrixXmat label_this_batch(batch_size,nClasses);
vector<vector<MatrixXmat> > convolution_input;
vector<vector<MatrixXmat> > convolution_output;
vector<vector<MatrixXmat> > pooling_output;
MatrixXmat linear_input;
MatrixXmat linear_output;
MatrixXi prediction_output;
//containers for the backward routine
MatrixXmat back_prop_out_layer;
MatrixXmat back_prop_linear_layer;
vector<vector<MatrixXmat> > back_prop_conv_flat;
vector<vector<MatrixXmat> > back_prop_pooling_2;
vector<vector<MatrixXmat> > back_prop_conv_layer_2;
vector<vector<MatrixXmat> > back_prop_pooling_1;
vector<vector<MatrixXmat> > back_prop_conv_layer_1;  //only for containing
//test the accuracy
MatrixXmat one_hot_prediction_train;
MatrixXmat one_hot_prediction_test;
UnitType train_accuracy;
UnitType test_accuracy;
UnitType prev_test_accuracy;
clock_t t_batch_start,t_batch_end;
//inference and train
for(unsigned int i=0;i<max_interation;i++){
    //perform stochastic gradient descent
    cout<<"Carrying out the "<<i+1<<" iteration"<<endl;
    prev_test_accuracy = test_accuracy;
    unsigned int batch_numbers = train_amount/batch_size;
    UnitType itr_loss = 0;
    for(unsigned int j=0;j<batch_numbers;j++){
    		if(j==0){
    			t_batch_start=clock();
    			}
        srand( (unsigned)time( NULL ));
        index_this_batch = stochastic_index(train_amount,batch_size);
        data_this_batch = stochastic_batch_data_sampling(data_train,index_this_batch);
        label_this_batch = stochastic_batch_label_sampling(label_train,index_this_batch);
        /*********Tranform data**********/
        convolution_input = data_squaring(data_this_batch,28,28,1);
        /*****************Forward routine***************/
        //cout<<"1"<<endl;
        //Conv 1
        convolution_output = conv_layer_1.forward_func(convolution_input);
        //cout<<"2"<<endl;
        //pooling
        //pooling_output = pool_layer_1.forward_func(convolution_output);
        //cout<<"3"<<endl;
         //Conv 2
        convolution_output = conv_layer_2.forward_func(convolution_output);
        //cout<<"4"<<endl;
        //pooling
        //pooling_output = pool_layer_2.forward_func(convolution_output);
        //cout<<"5"<<endl;
        //flatten the data
        linear_input = data_flatting(convolution_output);
        //cout<<"6"<<endl;
        linear_output = fully_connected_layer.forward_func(linear_input);
        //cout<<"7"<<endl;
        //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
        prediction_output = output_layer.output_func(linear_output,label_this_batch);
        itr_loss += output_layer.loss;
        /************Backward Routine*************/
        //loss layer
        back_prop_out_layer = output_layer.backward_func();
        back_prop_linear_layer = fully_connected_layer.backward_func(back_prop_out_layer);
        //cout<<"8"<<endl;
        //train
        fully_connected_layer.train_func(opt_trans_layer,"None",0.01);
        //transfer the gradient
        //[nData * 9 * 7 * 7]
        back_prop_conv_flat = data_squaring(back_prop_linear_layer,28,28,9);
        //cout<<"9"<<endl;
        //back_prop_pooling_2 = pool_layer_2.backward_func(back_prop_conv_flat);
        back_prop_conv_layer_2 = conv_layer_2.backward_func(back_prop_conv_flat);
        //cout<<"10"<<endl;
        //back_prop_pooling_1  = pool_layer_1.backward_func(back_prop_conv_layer_2);
        back_prop_conv_layer_1 = conv_layer_1.backward_func(back_prop_conv_layer_2);
        //cout<<"11"<<endl;
        conv_layer_1.train(opt_convolutional_layer_1,"None",0.01);
        conv_layer_2.train(opt_convolutional_layer_2,"None",0.01);
        cout<<(double)(output_layer.loss)<<endl;
        if(j==0){
	    			 	t_batch_end=clock();
					    float diff_train_batch = ((float)t_batch_end-(float)t_batch_start);
					    float seconds_train_batch = diff_train_batch / CLOCKS_PER_SEC;
					    cout<<"Single Batch Running Time is:"<<seconds_train_batch<<endl;
    				}
        }
    //print loss
    cout<<"The trainning of the "<<i+1<<" iteration has finished!"<<endl;
    cout<<"The loss of the "<<i+1<<" iteration is: "<<(double)(itr_loss)<<endl;
    clock_t t1,t2;
    t1=clock();
    /******Calculate the training accuracy for all the data*******/
    convolution_input = data_squaring(data_train,28,28,1);
    //layer 1
    convolution_output = conv_layer_1.forward_func(convolution_input);
    //pooling
    //pooling_output = pool_layer_1.forward_func(convolution_output);
    //layer2
    convolution_output = conv_layer_2.forward_func(convolution_output);
    //pooling
    //pooling_output = pool_layer_2.forward_func(convolution_output);
    //flatten the data
    linear_input = data_flatting(convolution_output);
    linear_output = fully_connected_layer.forward_func(linear_input);
    //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
    prediction_output = output_layer.output_func(linear_output,label_train);
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
    t2=clock();
    float diff_train = ((float)t2-(float)t1);
    float seconds_train = diff_train / CLOCKS_PER_SEC;
    cout<<"The training data Inference time is:"<<seconds_train<<endl;
    /**************calculste the test accuracy****************/
    clock_t t3,t4;
    t3=clock();
    convolution_input = data_squaring(data_test,28,28,1);
    //layer 1
    convolution_output = conv_layer_1.forward_func(convolution_input);
    //pooling
    //pooling_output = pool_layer_1.forward_func(convolution_output);
    //layer2
    convolution_output = conv_layer_2.forward_func(convolution_output);
    //pooling
    //pooling_output = pool_layer_2.forward_func(convolution_output);
    //flatten the data
    linear_input = data_flatting(convolution_output);
    linear_output = fully_connected_layer.forward_func(linear_input);
    //prediction_output is in shape [m_amount * 1] (not one_hot encoding)
    prediction_output = output_layer.output_func(linear_output,label_test);
    one_hot_prediction_test = one_hot_prediction_encoding(prediction_output,nClasses);
    test_accuracy = output_layer.acc_classfication(one_hot_prediction_test,label_test);
    t4=clock();
    float diff_test = ((float)t4-(float)t3);
    float seconds_test = diff_test / CLOCKS_PER_SEC;
    cout<<"The training data Inference time is:"<<seconds_test<<endl;
    if(test_accuracy>=0.945&&test_accuracy<=prev_test_accuracy){
        string file_path = "paramters_text/CNN_without_pooling/";
        Save_mat(conv_layer_1.Para_copy_store(),file_path+"W_conv_1.txt");
        Save_mat(conv_layer_2.Para_copy_store(),file_path+"W_conv_2.txt");
        Save_mat(fully_connected_layer.Para_copy_store(),file_path+"W_1.txt");
        break;
        }
    //print out the information
    cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
    cout<<"The Test accuracy of the "<<i+1<<" iteration is: "<<(double)(test_accuracy)<<endl;
    }
    cout<<"CNN program finished and the parameters have been saved!"<<endl;
}


//Test a demo of RNN model
void fun_rnn(){
    /***Recurrent Network Layer Initialization Instruction***
    *The network should be [nInput_dim * n_Rnn_size * n_output_dim]
    *The data should be reshaped into [nData * nLength * nDim] with the corresponding function
    ******************************************************/
    int train_amount = 10000;
    int test_amount = 1000;
    //read the data
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test = pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    unsigned int max_interation = 100;
    int batch_size = 256;
    double learning_rate = 1e-4;
    int rnn_size = 128;
    int n_input_Dim = 28;
    int n_input_length = 28;
    //Construct Layers
    //cout<<"1"<<endl;
    basic_rnn rnn_layer(n_input_Dim,rnn_size,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //Optimizer
    //sgd_optimizer rnn_optimizer(learning_rate,3);
    Adam_optimizer<UnitType> rnn_optimizer(learning_rate,3,0.9,0.999,1e-9);
    //define the containers to process the data
    //container of input
    vector<int> index_this_batch(batch_size);
    MatrixXmat data_this_batch;
    MatrixXmat label_this_batch;
    vector<MatrixXmat> current_data_input;  //nData * nLength * n_input_dim
    vector<MatrixXmat> current_data_output;   //nData * nLength * n_output_dim
    MatrixXmat linear_output;
    MatrixXi prediction_output;
    //containers for the backward routine
    MatrixXmat back_prop_out_layer;
    vector<MatrixXmat> back_prop_out_processed;
    vector<int> back_prop_label(1);
    back_prop_label[0] = n_input_length;
    //test the accuracy
    MatrixXmat one_hot_prediction_train;
    MatrixXmat one_hot_prediction_test;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    clock_t t_batch_start,t_batch_end;
    for(unsigned int i=0;i<max_interation;i++){
        //perform stochastic gradient descent
        cout<<"Carrying out the "<<i+1<<" iteration"<<endl;
        prev_test_accuracy = test_accuracy;
        unsigned int batch_numbers = train_amount/batch_size;
        UnitType itr_loss = 0;
        for(unsigned int j=0;j<batch_numbers;j++){
	        	if(j==0){
	    				t_batch_start=clock();
	    			}
            srand( (unsigned)time( NULL ));
            index_this_batch = stochastic_index(train_amount,batch_size);
            data_this_batch = stochastic_batch_data_sampling(data_train,index_this_batch);
            label_this_batch = stochastic_batch_label_sampling(label_train,index_this_batch);
            //cout<<"3"<<endl;
            /*********Tranform data**********/
            current_data_input = input_reshape_rnn(data_this_batch,n_input_length,n_input_Dim);
            //cout<<"4"<<endl;
            /*********Forward RNN**********/
            current_data_output = rnn_layer.forward_func(current_data_input);
            linear_output = ind_data_exact(current_data_output,n_input_length);
            //cout<<"5"<<endl;
            /*********Transform into labels********/
            prediction_output = output_layer.output_func(linear_output,label_this_batch);
            //add the loss
            itr_loss += output_layer.loss;
            back_prop_out_layer = output_layer.backward_func();    //[nData * nDim]
            back_prop_out_processed = input_reshape_rnn(back_prop_out_layer,1,nClasses);
            //cout<<"6"<<endl;
            rnn_layer.gradient_func(back_prop_out_processed,back_prop_label);
            //cout<<"7"<<endl;
            rnn_layer.train(rnn_optimizer);
            cout<<(double)(output_layer.loss)<<endl;
            if(j==0){
	    			 	t_batch_end=clock();
					    float diff_train_batch = ((float)t_batch_end-(float)t_batch_start);
					    float seconds_train_batch = diff_train_batch / CLOCKS_PER_SEC;
					    cout<<"Single Batch Running Time is:"<<seconds_train_batch<<endl;
    				}
        }
        //print loss
        cout<<"The trainning of the "<<i+1<<" iteration has finished!"<<endl;
        cout<<"The loss of the "<<i+1<<" iteration is: "<<(double)(itr_loss)<<endl;
        clock_t t1,t2;
    		t1=clock();
        /**********Compute the train accuracy*********/
        current_data_input = input_reshape_rnn(data_train,n_input_length,n_input_Dim);
        current_data_output = rnn_layer.forward_func(current_data_input);
        linear_output = ind_data_exact(current_data_output,n_input_length);
        prediction_output = output_layer.output_func(linear_output,label_train);
        one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
        train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
        t2=clock();
		    float diff_train = ((float)t2-(float)t1);
		    float seconds_train = diff_train / CLOCKS_PER_SEC;
		    cout<<"The training data Inference time is:"<<seconds_train<<endl;
		    /**************calculste the test accuracy****************/
		    clock_t t3,t4;
		    t3=clock();
        current_data_input = input_reshape_rnn(data_test,n_input_length,n_input_Dim);
        current_data_output = rnn_layer.forward_func(current_data_input);
        linear_output = ind_data_exact(current_data_output,n_input_length);
        prediction_output = output_layer.output_func(linear_output,label_test);
        one_hot_prediction_test = one_hot_prediction_encoding(prediction_output,nClasses);
        test_accuracy = output_layer.acc_classfication(one_hot_prediction_test,label_test);
        t4=clock();
	      float diff_test = ((float)t4-(float)t3);
	      float seconds_test = diff_test / CLOCKS_PER_SEC;
	      cout<<"The Test data Inference time is:"<<seconds_test<<endl;
        //print out the information
        cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
        cout<<"The Test accuracy of the "<<i+1<<" iteration is: "<<(double)(test_accuracy)<<endl;
        if(train_accuracy>=0.85&&test_accuracy<=prev_test_accuracy){
            vector<MatrixXmat> para_container;
            para_container = rnn_layer.Para_copy_store();
            string file_path = "paramters_text/RNN/";
            Save_mat(para_container[0],file_path+"W_input.txt");
            Save_mat(para_container[1],file_path+"U_trans.txt");
            Save_mat(para_container[2],file_path+"V_out.txt");
            break;
            }
        else if(i==max_interation-1){
            vector<MatrixXmat> para_container;
            para_container = rnn_layer.Para_copy_store();
            string file_path = "paramters_text/RNN/";
            Save_mat(para_container[0],file_path+"W_input.txt");
            Save_mat(para_container[1],file_path+"U_trans.txt");
            Save_mat(para_container[2],file_path+"V_out.txt");
        }
    }
    cout<<"The program has been finished and the parameters has been saved!"<<endl;
}

//Test a demo of LSTM model
void fun_lstm(){
    /***Recurrent Network Layer Initialization Instruction***
    *The network should be [nInput_dim * n_Rnn_size * n_output_dim]
    *The data should be reshaped into [nData * nLength * nDim] with the corresponding function
    ******************************************************/
    int train_amount = 10000;
    int test_amount = 1000;
    //read the data
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test = pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    unsigned int max_interation = 100;
    int batch_size = 256;
    UnitType learning_rate = 1e-6;
    int rnn_size = 128;
    int n_input_Dim = 28;
    int n_input_length = 28;
    //Construct Layers
    //cout<<"1"<<endl;
    lstm_rnn rnn_layer(n_input_Dim,rnn_size,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //Optimizer
//    Adam_optimizer opt_trans_layer(learning_rate,1,0.9,0.999,1e-8);
//    Adam_optimizer opt_convolutional_layer(learning_rate,3,0.9,0.999,1e-8);
    sgd_optimizer<UnitType> rnn_optimizer(learning_rate,9);
    //Adam_optimizer<UnitType> rnn_optimizer(learning_rate,9,0.9,0.999,1e-8);   //9 parameters to learn
    //cout<<"2"<<endl;
    //define the containers to process the data
    //container of input
    vector<int> index_this_batch(batch_size);
    MatrixXmat data_this_batch;
    MatrixXmat label_this_batch;
    vector<MatrixXmat> current_data_input;  //nData * nLength * n_input_dim
    vector<MatrixXmat> current_data_output;   //nData * nLength * n_output_dim
    MatrixXmat linear_output;
    MatrixXi prediction_output;
    //containers for the backward routine
    MatrixXmat back_prop_out_layer;
    vector<MatrixXmat> back_prop_out_processed;
    vector<int> back_prop_label(1);
    back_prop_label[0] = n_input_length;
    //test the accuracy
    MatrixXmat one_hot_prediction_train;
    MatrixXmat one_hot_prediction_test;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    clock_t t_batch_start,t_batch_end;
    for(unsigned int i=0;i<max_interation;i++){
        //perform stochastic gradient descent
        cout<<"Carrying out the "<<i+1<<" iteration"<<endl;
        prev_test_accuracy = test_accuracy;
        unsigned int batch_numbers = train_amount/batch_size;
        UnitType itr_loss = 0;
        for(unsigned int j=0;j<batch_numbers;j++){
        	 if(j==0){
	    				t_batch_start=clock();
	    			}
            srand( (unsigned)time( NULL ));
            index_this_batch = stochastic_index(train_amount,batch_size);
            data_this_batch = stochastic_batch_data_sampling(data_train,index_this_batch);
            label_this_batch = stochastic_batch_label_sampling(label_train,index_this_batch);
            //cout<<"3"<<endl;
            /*********Tranform data**********/
            current_data_input = input_reshape_rnn(data_this_batch,n_input_length,n_input_Dim);
            //cout<<"4"<<endl;
            /*********Forward RNN**********/
            current_data_output = rnn_layer.forward_func(current_data_input);
            //cout<<"zjr"<<endl;
            linear_output = ind_data_exact(current_data_output,n_input_length);
            //cout<<"5"<<endl;
            /*********Transform into labels********/
            prediction_output = output_layer.output_func(linear_output,label_this_batch);
            //add the loss
            itr_loss += output_layer.loss;
            //cout<<"55"<<endl;
            back_prop_out_layer = output_layer.backward_func();    //[nData * nDim]
            back_prop_out_processed = input_reshape_rnn(back_prop_out_layer,1,nClasses);
            //cout<<"6"<<endl;
            rnn_layer.gradient_func(back_prop_out_processed,back_prop_label);
            //cout<<"7"<<endl;
            rnn_layer.train(rnn_optimizer);
            //cout<<"8"<<endl;
            cout<<(double)(output_layer.loss)<<endl;
            if(j==0){
	    			 	t_batch_end=clock();
					    float diff_train_batch = ((float)t_batch_end-(float)t_batch_start);
					    float seconds_train_batch = diff_train_batch / CLOCKS_PER_SEC;
					    cout<<"Single Batch Running Time is:"<<seconds_train_batch<<endl;
    				}
    				break;
        }
        //print loss
        cout<<"The trainning of the "<<i+1<<" iteration has finished!"<<endl;
        cout<<"The loss of the "<<i+1<<" iteration is: "<<(double)(itr_loss)<<endl;
        clock_t t1,t2;
    		t1=clock();
        /**********Compute the train accuracy*********/
        current_data_input = input_reshape_rnn(data_train,n_input_length,n_input_Dim);
        current_data_output = rnn_layer.forward_func(current_data_input);
        linear_output = ind_data_exact(current_data_output,n_input_length);
        prediction_output = output_layer.output_func(linear_output,label_train);
        one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
        train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
        //print out the information
        cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
        t2=clock();
		    float diff_train = ((float)t2-(float)t1);
		    float seconds_train = diff_train / CLOCKS_PER_SEC;
		    cout<<"The training data Inference time is:"<<seconds_train<<endl;
		    /**************calculste the test accuracy****************/
		    clock_t t3,t4;
		    t3=clock();
        current_data_input = input_reshape_rnn(data_test,n_input_length,n_input_Dim);
        current_data_output = rnn_layer.forward_func(current_data_input);
        linear_output = ind_data_exact(current_data_output,n_input_length);
        prediction_output = output_layer.output_func(linear_output,label_test);
        one_hot_prediction_test = one_hot_prediction_encoding(prediction_output,nClasses);
        test_accuracy = output_layer.acc_classfication(one_hot_prediction_test,label_test);
        t4=clock();
	      float diff_test = ((float)t4-(float)t3);
	      float seconds_test = diff_test / CLOCKS_PER_SEC;
	      cout<<"The Test data Inference time is:"<<seconds_test<<endl;
        //print out the information
        cout<<"The Train accuracy of the "<<i+1<<" iteration is: "<<(double)(train_accuracy)<<endl;
        cout<<"The Test accuracy of the "<<i+1<<" iteration is: "<<(double)(test_accuracy)<<endl;
    }
}

int main(){
    srand( (unsigned)time( NULL ));
    fun_dnn();
    //fun_dnn();
    //fun_cnn();
    //fun_cnn_without_pool();
    //fun_rnn();
    //fun_lstm();
    cout<<"Enter any key to continue..."<<endl;
    char aa;
    cin>>aa;
}
