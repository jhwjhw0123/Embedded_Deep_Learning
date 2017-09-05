/**************************
*@description: This is a file that demonstrate the usage of the library
*@usage: run the code with other source and head files
*@author: Chen Wang, Dept. of Computer Science, University College London
*@version: 0.01
***************************/

/*******This demo shows a three-hidden layer neural network********
layer1: linear 1000 nodes + RELU
layer2: linear 500 nodes + RELU
layer3: linear 100 nodes + RELU
output_layer: softmax cross-entropy layer
*****************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <cstdlib>
#define PI 3.1415927
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "DNN.h"
#include "CNN.h"
#include "RNN.h"
#include "MatrixOperation.h"
#include "MathType.h"
#include "mnist/mnist_reader.hpp"
#include "jsonLib/json.hpp"
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

void fun_dnn_node_prune_10(){
    //Set how many data to be used, change here is you want
    int train_amount = 10000;
    int test_amount = 300;
    //read the data
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    cout<<"Reading data..."<<endl;
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test= pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    cout<<"Data successfully loaded..."<<endl;
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    //Consruct the layers
    Linear_Layer layer_1_trans(nDim,1000);
    RELU_Layer layer_1_activ(1000);
    Linear_Layer layer_2_trans(1000,500);
    RELU_Layer layer_2_activ(500);
    Linear_Layer layer_3_trans(500,100);
    RELU_Layer layer_3_activ(100);
    Linear_Layer output_trans_layer(100,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //Assign value
    string file_path = "paramters_text/DNN/";
    cout<<"Reading parameters..."<<endl;
    layer_1_trans.Para_copy_read(Load_mat(file_path+"W_1.txt",nDim+1,1000));
    layer_2_trans.Para_copy_read(Load_mat(file_path+"W_2.txt",1001,500));
    layer_3_trans.Para_copy_read(Load_mat(file_path+"W_3.txt",501,100));
    output_trans_layer.Para_copy_read(Load_mat(file_path+"W_4.txt",101,nClasses));
    cout<<"Parameters Reading finished!"<<endl;
    //The pruning path
    string pruning_path_layer = "node_pruning/layer_info/";
    string pruning_path_weight = "node_pruning/weight_info/";
    string save_path = "node_prune_10/";
    //define the containers to process the data
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
    MatrixXmat layer_out_input;
    MatrixXi prediction_output;
    MatrixXmat prediction_save;
    //Variables to compute accuracy
    MatrixXmat one_hot_prediction_train;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    cout<<"Making inference for train data..."<<endl;
    //Inference
    /***************************************************
    Prune the nodes of first layer
    ****************************************************/
    cout<<"Pruning the nodes of the first layer..."<<endl;
    /***************layer 1***************/
    //Save_mat(data_train,weight_pruning_path_layer+"prev_layer.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_1_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_1_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer1.py");
    /***************************************************
    Prune the nodes of second layer
    ****************************************************/
    cout<<"Pruning the nodes of the second layer..."<<endl;
    /***************layer 1***************/
    layer_1_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",nDim+1,1000));
    Save_mat(layer_1_trans.Para_copy_store(),save_path+"W_1_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_2_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_2_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer2.py");
    /***************************************************
    Prune the nodes of third layer
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_2_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",1001,500));
    Save_mat(layer_2_trans.Para_copy_store(),save_path+"W_2_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_3_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_3_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer3.py");
    /***************************************************
    Final Inference
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_3_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",501,100));
    Save_mat(layer_3_trans.Para_copy_store(),save_path+"W_3_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
    //print out the information
    cout<<"The Train accuracy of the network is: "<<(float)(train_accuracy)<<endl;
    /******Calculate the training accuracy for all the data*******/
    cout<<"Making inference for test data..."<<endl;
    //layer 1
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
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    test_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_test);
    cout<<"The Test accuracy of the network is: "<<(float)(test_accuracy)<<endl;
    }

void fun_dnn_node_prune_30(){
    //Set how many data to be used, change here is you want
    int train_amount = 10000;
    int test_amount = 300;
    //read the data
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    cout<<"Reading data..."<<endl;
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test= pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    cout<<"Data successfully loaded..."<<endl;
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    //Consruct the layers
    Linear_Layer layer_1_trans(nDim,1000);
    RELU_Layer layer_1_activ(1000);
    Linear_Layer layer_2_trans(1000,500);
    RELU_Layer layer_2_activ(500);
    Linear_Layer layer_3_trans(500,100);
    RELU_Layer layer_3_activ(100);
    Linear_Layer output_trans_layer(100,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //Assign value
    string file_path = "paramters_text/DNN/";
    cout<<"Reading parameters..."<<endl;
    layer_1_trans.Para_copy_read(Load_mat(file_path+"W_1.txt",nDim+1,1000));
    layer_2_trans.Para_copy_read(Load_mat(file_path+"W_2.txt",1001,500));
    layer_3_trans.Para_copy_read(Load_mat(file_path+"W_3.txt",501,100));
    output_trans_layer.Para_copy_read(Load_mat(file_path+"W_4.txt",101,nClasses));
    cout<<"Parameters Reading finished!"<<endl;
    //The pruning path
    string pruning_path_layer = "node_pruning/layer_info/";
    string pruning_path_weight = "node_pruning/weight_info/";
    string save_path = "node_prune_30/";
    //define the containers to process the data
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
    MatrixXmat layer_out_input;
    MatrixXi prediction_output;
    MatrixXmat prediction_save;
    //Variables to compute accuracy
    MatrixXmat one_hot_prediction_train;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    cout<<"Making inference for train data..."<<endl;
    //Inference
    /***************************************************
    Prune the nodes of first layer
    ****************************************************/
    cout<<"Pruning the nodes of the first layer..."<<endl;
    /***************layer 1***************/
    //Save_mat(data_train,weight_pruning_path_layer+"prev_layer.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_1_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_1_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer1_30.py");
    /***************************************************
    Prune the nodes of second layer
    ****************************************************/
    cout<<"Pruning the nodes of the second layer..."<<endl;
    /***************layer 1***************/
    layer_1_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",nDim+1,1000));
    Save_mat(layer_1_trans.Para_copy_store(),save_path+"W_1_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_2_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_2_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer2_30.py");
    /***************************************************
    Prune the nodes of third layer
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_2_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",1001,500));
    Save_mat(layer_2_trans.Para_copy_store(),save_path+"W_2_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_3_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_3_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer3_30.py");
    /***************************************************
    Final Inference
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_3_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",501,100));
    Save_mat(layer_3_trans.Para_copy_store(),save_path+"W_3_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
    //print out the information
    cout<<"The Train accuracy of the network is: "<<(float)(train_accuracy)<<endl;
    /******Calculate the training accuracy for all the data*******/
    cout<<"Making inference for test data..."<<endl;
    //layer 1
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
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    test_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_test);
    cout<<"The Test accuracy of the network is: "<<(float)(test_accuracy)<<endl;
    }

void fun_dnn_node_prune_50(){
    //Set how many data to be used, change here is you want
    int train_amount = 10000;
    int test_amount = 300;
    //read the data
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    cout<<"Reading data..."<<endl;
    //train data in [n_trainData * nDim] shape
    MatrixXmat data_train = pre_process_images<UnitType>(dataset.training_images,train_amount);
    //test data in [n_testData * nDim] shape
    MatrixXmat data_test= pre_process_images<UnitType>(dataset.test_images,test_amount);
    //train labels in [n_trainData * 1] shape
    MatrixXmat label_train = pre_process_labels(dataset.training_labels,train_amount,10);
    //test labels in [n_testData * 1] shape
    MatrixXmat label_test = pre_process_labels(dataset.test_labels,test_amount,10);
    cout<<"Data successfully loaded..."<<endl;
    //Get hyper-parameters
    int nDim = data_train.cols();
    int nClasses = 10;
    //Consruct the layers
    Linear_Layer layer_1_trans(nDim,1000);
    RELU_Layer layer_1_activ(1000);
    Linear_Layer layer_2_trans(1000,500);
    RELU_Layer layer_2_activ(500);
    Linear_Layer layer_3_trans(500,100);
    RELU_Layer layer_3_activ(100);
    Linear_Layer output_trans_layer(100,nClasses);
    soft_max_cross_entropy_layer<UnitType> output_layer(nClasses);
    //Assign value
    string file_path = "paramters_text/DNN/";
    cout<<"Reading parameters..."<<endl;
    layer_1_trans.Para_copy_read(Load_mat(file_path+"W_1.txt",nDim+1,1000));
    layer_2_trans.Para_copy_read(Load_mat(file_path+"W_2.txt",1001,500));
    layer_3_trans.Para_copy_read(Load_mat(file_path+"W_3.txt",501,100));
    output_trans_layer.Para_copy_read(Load_mat(file_path+"W_4.txt",101,nClasses));
    cout<<"Parameters Reading finished!"<<endl;
    //The pruning path
    string pruning_path_layer = "node_pruning/layer_info/";
    string pruning_path_weight = "node_pruning/weight_info/";
    string save_path = "node_prune_50/";
    //define the containers to process the data
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
    MatrixXmat layer_out_input;
    MatrixXi prediction_output;
    MatrixXmat prediction_save;
    //Variables to compute accuracy
    MatrixXmat one_hot_prediction_train;
    UnitType train_accuracy = 0;
    UnitType test_accuracy = 0;
    UnitType prev_test_accuracy = 0;
    cout<<"Making inference for train data..."<<endl;
    //Inference
    /***************************************************
    Prune the nodes of first layer
    ****************************************************/
    cout<<"Pruning the nodes of the first layer..."<<endl;
    /***************layer 1***************/
    //Save_mat(data_train,weight_pruning_path_layer+"prev_layer.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_1_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_1_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer1_50.py");
    /***************************************************
    Prune the nodes of second layer
    ****************************************************/
    cout<<"Pruning the nodes of the second layer..."<<endl;
    /***************layer 1***************/
    layer_1_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",nDim+1,1000));
    Save_mat(layer_1_trans.Para_copy_store(),save_path+"W_1_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_2_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_2_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer2_50.py");
    /***************************************************
    Prune the nodes of third layer
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_2_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",1001,500));
    Save_mat(layer_2_trans.Para_copy_store(),save_path+"W_2_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    Save_mat(layer_3_output,pruning_path_layer+"layer.txt");
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    //cast the output to save
    prediction_save = prediction_output.cast<UnitType>();
    Save_mat(prediction_save,pruning_path_layer+"output.txt");
    Save_mat(layer_3_trans.Para_copy_store(),pruning_path_weight+"Weight_this_layer.txt");
    system("Python3 node_pruning_layer3_50.py");
    /***************************************************
    Final Inference
    ****************************************************/
    cout<<"Pruning the nodes of the third layer..."<<endl;
    /***************layer 1***************/
    layer_3_trans.Para_copy_read(Load_mat(pruning_path_weight+"Weight_node_pruned.txt",501,100));
    Save_mat(layer_3_trans.Para_copy_store(),save_path+"W_3_Node_pruned.txt");
    layer_1_input = layer_1_trans.forward_func(data_train);
    layer_1_output = layer_1_activ.forward_func(layer_1_input);
    /****************layer 2***************/
    //Save_mat(layer_1_output,weight_pruning_path_layer+"prev_layer.txt");
    layer_2_input = layer_2_trans.forward_func(layer_1_output);
    layer_2_output = layer_2_activ.forward_func(layer_2_input);
    /****************layer 3***************/
    layer_3_input = layer_3_trans.forward_func(layer_2_output);
    layer_3_output = layer_3_activ.forward_func(layer_3_input);
    /*************output layer*************/
    layer_out_input = output_trans_layer.forward_func(layer_3_output);
    prediction_output = output_layer.output_func(layer_out_input,label_train);
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    train_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_train);
    //print out the information
    cout<<"The Train accuracy of the network is: "<<(float)(train_accuracy)<<endl;
    /******Calculate the training accuracy for all the data*******/
    cout<<"Making inference for test data..."<<endl;
    //layer 1
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
    cout<<"Calculating the accuracy..."<<endl;
    one_hot_prediction_train = one_hot_prediction_encoding(prediction_output,nClasses);
    test_accuracy = output_layer.acc_classfication(one_hot_prediction_train,label_test);
    cout<<"The Test accuracy of the network is: "<<(float)(test_accuracy)<<endl;
    }

int main(){
    srand( (unsigned)time( NULL ));
    //fun_dnn_node_prune_10();
    fun_dnn_node_prune_30();
    fun_dnn_node_prune_50();
    //fun_cnn_inference();
    //fun_rnn();
    //fun_lstm();
     //If sufficiently high test accuracy and start to overfit
    cout<<"Program Finished! Press any key + enter to finish the program."<<endl;
    char a;
    cin>>a;
}
