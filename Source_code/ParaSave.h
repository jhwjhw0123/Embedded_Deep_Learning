#ifndef PARASAVE_H_INCLUDED
#define PARASAVE_H_INCLUDED
#include "MathType.h"
#include <vector>
// #include "jsonLib/json.hpp"
#include <fstream>

using namespace std;
using namespace Eigen;

/************Codes for DNN**************/
//Serialization
vector<vector<UnitType> > sqaure_serialize(MatrixXmat mat);
//parameter saving
void para_save_DNN(vector<MatrixXmat> para, string file_name);
MatrixXmat para_read_2d(string file_name, string var_name);

/************Codes for CNN************/
//Serialization
vector<UnitType> conv_serialize(MatrixXmat mat);
//parameter saving
void para_save_CNN(vector<vector<vector<MatrixXmat> > > para_conv, vector<MatrixXmat> para_full,string file_name);
vector<vector<MatrixXmat> > para_read_conv(string file_name, string var_name);

/************Codes for RNN*************/
void para_save_RNN(vector<MatrixXmat> para_rnn, vector<MatrixXmat> para_2d, string file_name);
void para_save_LSTM(vector<MatrixXmat> para_lstm, string file_name);
vector<MatrixXmat> rnn_para_read(string file_name);

/************save to text file heads*****************/
void Save_mat(MatrixXmat mat, string file_name);

MatrixXmat Load_mat(string file_name, int row, int col);

void Save_mat(vector<vector<MatrixXmat > > mat, string file_name);

vector<vector<MatrixXmat > > Load_mat(string file_name, int vecRow, int vecCol, int row, int col);

#endif // PARASAVE_H_INCLUDED
