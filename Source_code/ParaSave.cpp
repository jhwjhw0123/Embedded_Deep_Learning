#include "ParaSave.h"
#include "MathType.h"
#include <stdlib.h>
#include <iostream>
#include <exception>
#include "Eigen/Dense"
#include "Eigen/Sparse"
// #include "jsonLib/json.hpp"

using namespace std;
using namespace Eigen;
// using json = nlohmann::json;

// //DNN
// //square parameter serializationn
// vector<vector<UnitType> > sqaure_serialize(MatrixXmat mat){
//     int nDim = mat.rows();
//     int nNode = mat.cols();
//     vector<vector<UnitType> > serial_vec(nDim,vector<UnitType>(nNode));
//     for(unsigned int i=0;i<nDim;i++){
//         for(unsigned int j=0;j<nNode;j++){
//             serial_vec[i][j] = mat(i,j);
//         }
//     }

//     return serial_vec;
// }

// //save parameters
// void para_save_DNN(vector<MatrixXmat> para,string file_name){
//     //get the layer numbers
//     int nLayer = para.size();
//     json jsonfile;
//     string base_str = "W_";
//     vector<vector<vector<UnitType> > > current_layer_series(nLayer);
//     string current_string;
//     for(unsigned int i=0;i<nLayer;i++){
//         current_layer_series[i] = sqaure_serialize(para[i]);
//     }
//     for(unsigned int i=0;i<nLayer;i++){
//         current_string = base_str+to_string(i+1);
//         jsonfile.emplace(current_string,current_layer_series[i]);
//     }
//     ofstream file(file_name);
//     file << jsonfile;
// }

// //parameter read
// MatrixXmat para_read_2d(string file_name, string var_name){
//     ifstream ifs(file_name);
//     json j_read = json::parse(ifs);
//     vector<vector<UnitType> > para_read = j_read[var_name].get<vector<vector<UnitType> > >();
//     int nRow = para_read.size();
//     int nColumn = para_read[0].size();
//     MatrixXmat rst_mat(nRow,nColumn);
//     for(unsigned int i=0;i<nRow;i++){
//         for(unsigned int j=0;j<nColumn;j++){
//             rst_mat(i,j) = para_read[i][j];
//         }
//     }

//     return rst_mat;
// }

// //CNN
// //Serialization
// vector<vector<vector<vector<UnitType> > > > conv_serialize(vector<vector<MatrixXmat> > mat){
//     //mat should be in [input_channel * output_channel * height * width]
//     int input_channel = mat.size();
//     int output_channel = mat[0].size();
//     int height = mat[0][0].rows();
//     int width = mat[0][0].cols();
//     vector<vector<vector<vector<UnitType> > > > rst_mat(input_channel,vector<vector<vector<UnitType> > >(output_channel));
//     vector<vector<UnitType> > current_conv;
//     for(unsigned int i=0;i<input_channel;i++){
//         for(unsigned int j=0;j<output_channel;j++){
//             current_conv = sqaure_serialize(mat[i][j]);
//             rst_mat[i][j] = current_conv;
//         }
//     }

//     return rst_mat;
// }

// //parameter saving
// void para_save_CNN(vector<vector<vector<MatrixXmat> > > para_conv, vector<MatrixXmat> para_full, string file_name){
//     int nConv = para_conv.size();
//     int nFull = para_full.size();
//     json jsonfile;
//     string conv_str_base = "W_conv_";
//     string full_str_base = "W_";
//     string current_string;
//     vector<vector<vector<vector<vector<UnitType> > > > > conv_series_data(nConv);
//     vector<vector<vector<UnitType> > > full_series_data(nFull);
//     //convolve parameters
//     for(unsigned int i=0;i<nConv;i++){
//         conv_series_data[i] = conv_serialize(para_conv[i]);
//     }
//     for(unsigned int i=0;i<nConv;i++){
//         current_string = conv_str_base + to_string(i+1);
//         jsonfile.emplace(current_string,conv_series_data[i]);
//     }
//     //fully-connected parameters
//     for(unsigned int i=0;i<nFull;i++){
//         full_series_data[i] = sqaure_serialize(para_full[i]);
//     }
//     for(unsigned int i=0;i<nFull;i++){
//         current_string = full_str_base + to_string(i+1);
//         jsonfile.emplace(current_string,full_series_data[i]);
//     }
//     //save
//     ofstream file(file_name);
//     file << jsonfile;
// }

// //parameter read
// vector<vector<MatrixXmat> > para_read_conv(string file_name, string var_name){
//     ifstream ifs(file_name);
//     json j_read = json::parse(ifs);
//     vector<vector<vector<vector<UnitType> > > > para_read = j_read[var_name].get<vector<vector<vector<vector<UnitType> > > > >();
//     int input_channel = para_read.size();
//     int output_channel = para_read[0].size();
//     int height = para_read[0][0].size();
//     int width = para_read[0][0][0].size();
//     vector<vector<MatrixXmat> > rst_mat(input_channel,vector<MatrixXmat>(output_channel));
//     for(unsigned int i=0;i<input_channel;i++){
//         for(unsigned int j=0;j<output_channel;j++){
//             vector<vector<UnitType> > temp_serial_mat = para_read[i][j];
//             MatrixXmat temp_mat(height,width);
//             for(unsigned int k=0;k<height;k++){
//                 for(unsigned int p=0;p<height;p++){
//                     temp_mat(k,p) = temp_serial_mat[k][p];
//                 }
//             }
//             rst_mat[i][j]= temp_mat;
//         }
//     }

//     return rst_mat;
// }


// //RNN
// void para_save_RNN(vector<MatrixXmat> para_rnn, vector<MatrixXmat> para_2d, string file_name){
//     //0: w_input; 1: u_transform; 2: v_output
//     int nPara = para_rnn.size();
//     int nPara_full = para_2d.size();
//     json jsonfile;
//     //There should be 3 parameters, if not, throw an exception
//     if(nPara!=3){
//         throw std::invalid_argument("The parameters of RNN should be 3!");
//     }
//     vector<vector<UnitType> > w_input = sqaure_serialize(para_rnn[0]);
//     vector<vector<UnitType> > u_trans = sqaure_serialize(para_rnn[1]);
//     vector<vector<UnitType> > v_output = sqaure_serialize(para_rnn[2]);
//     jsonfile = {{"W_input",w_input},{"U_trans",u_trans},{"V_output",v_output}};
//     string full_str_base = "W_";
//     string current_string;
//     vector<vector<vector<UnitType> > > full_series_data(nPara_full);
//     //fully-connected parameters
//     for(unsigned int i=0;i<nPara_full;i++){
//         full_series_data[i] = sqaure_serialize(para_2d[i]);
//     }
//     for(unsigned int i=0;i<nPara_full;i++){
//         current_string = full_str_base + to_string(i+1);
//         jsonfile.emplace(current_string,full_series_data[i]);
//     }
//     //save
//     ofstream file(file_name);
//     file << jsonfile;
// }

// vector<MatrixXmat> rnn_para_read(string file_name){
//     //0: w_input; 1: u_transform; 2: v_output
//     ifstream ifs(file_name);
//     json j_read = json::parse(ifs);
//     vector<vector<UnitType> > para_W = j_read["W_input"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_U = j_read["U_trans"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_V = j_read["V_output"].get<vector<vector<UnitType> > >();
//     int input_dim = para_W.size();
//     int hidden_dim = para_U.size();
//     int output_dim = para_V[0].size();
//     vector<MatrixXmat> rst_mat(3);
//     MatrixXmat W_mat(input_dim,hidden_dim);
//     MatrixXmat U_mat(hidden_dim,hidden_dim);
//     MatrixXmat V_mat(hidden_dim,output_dim);
//     for(unsigned int i=0;i<input_dim;i++){
//         for(unsigned int j=0;j<hidden_dim;j++){
//             W_mat(i,j) = para_W[i][j];
//             if(i==0){
//                 for(unsigned int k=0;k<hidden_dim;k++){
//                     U_mat(j,k) = para_U[j][k];
//                 }
//                 for(unsigned int k=0;k<output_dim;k++){
//                     V_mat(j,k) = para_V[j][k];
//                 }
//             }
//         }
//     }
//     rst_mat[0] = W_mat;
//     rst_mat[1] = U_mat;
//     rst_mat[2] = V_mat;

//     return rst_mat;
// }

// void para_save_LSTM(vector<MatrixXmat> para_lstm, string file_name){
//     //0: w_f; 1: b_f; 2:w_i; 3: b_i; 4: w_c; 5: b_c; 6: w_o; 7: b_o; 8: v_output
//     //0: w_input; 1: u_transform; 2: v_output
//     int nPara = para_lstm.size();
//     json jsonfile;
//     //There should be 3 parameters, if not, throw an exception
//     if(nPara!=9){
//         throw std::invalid_argument("The parameters of LSTM should be 9!");
//     }
//     vector<vector<UnitType> > w_f = sqaure_serialize(para_lstm[0]);
//     vector<vector<UnitType> > b_f = sqaure_serialize(para_lstm[1]);
//     vector<vector<UnitType> > w_i = sqaure_serialize(para_lstm[2]);
//     vector<vector<UnitType> > b_i = sqaure_serialize(para_lstm[3]);
//     vector<vector<UnitType> > w_c = sqaure_serialize(para_lstm[4]);
//     vector<vector<UnitType> > b_c = sqaure_serialize(para_lstm[5]);
//     vector<vector<UnitType> > w_o = sqaure_serialize(para_lstm[6]);
//     vector<vector<UnitType> > b_o = sqaure_serialize(para_lstm[7]);
//     vector<vector<UnitType> > v_output = sqaure_serialize(para_lstm[8]);
//     jsonfile = {{"W_f",w_f},{"b_f",b_f},{"W_i",w_i},\
//                 {"b_i",b_i},{"W_c",w_c},{"b_c",b_c},\
//                 {"W_o",w_o},{"b_o",b_o},{"V_output",v_output}};
//     //save
//     ofstream file(file_name);
//     file << jsonfile;
// }

// vector<MatrixXmat> lstm_para_read(string file_name){
//     //0: w_input; 1: u_transform; 2: v_output
//     ifstream ifs(file_name);
//     json j_read = json::parse(ifs);
//     vector<vector<UnitType> > para_w_f = j_read["W_f"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_b_f = j_read["b_f"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_w_i = j_read["W_i"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_b_i = j_read["b_i"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_w_c = j_read["W_c"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_b_c = j_read["b_c"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_w_o = j_read["W_o"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_b_o = j_read["b_o"].get<vector<vector<UnitType> > >();
//     vector<vector<UnitType> > para_V = j_read["V_output"].get<vector<vector<UnitType> > >();
//     int concate_dim = para_w_f.size();
//     int hidden_dim = para_V.size();
//     int output_dim = para_V[0].size();
//     vector<MatrixXmat> rst_mat(9);
//     MatrixXmat W_f(concate_dim,hidden_dim);
//     MatrixXmat b_f(1,hidden_dim);
//     MatrixXmat W_i(concate_dim,hidden_dim);
//     MatrixXmat b_i(1,hidden_dim);
//     MatrixXmat W_c(concate_dim,hidden_dim);
//     MatrixXmat b_c(1,hidden_dim);
//     MatrixXmat W_o(concate_dim,hidden_dim);
//     MatrixXmat b_o(1,hidden_dim);
//     MatrixXmat V_mat(hidden_dim,output_dim);
//     for(unsigned int i=0;i<concate_dim;i++){
//         for(unsigned int j=0;j<hidden_dim;j++){
//             W_f(i,j) = para_w_f[i][j];
//             W_i(i,j) = para_w_i[i][j];
//             W_c(i,j) = para_w_c[i][j];
//             W_o(i,j) = para_w_o[i][j];
//             if(i==0){
//                 b_f(i,j) = para_b_f[i][j];
//                 b_i(i,j) = para_b_i[i][j];
//                 b_c(i,j) = para_b_c[i][j];
//                 b_o(i,j) = para_b_o[i][j];
//                 for(unsigned int k=0;k<output_dim;k++){
//                     V_mat(j,k) = para_V[j][k];
//                 }
//             }
//         }
//     }
//     rst_mat[0] = W_f;
//     rst_mat[1] = b_f;
//     rst_mat[2] = W_i;
//     rst_mat[3] = b_i;
//     rst_mat[4] = W_c;
//     rst_mat[5] = b_c;
//     rst_mat[6] = W_o;
//     rst_mat[7] = b_o;
//     rst_mat[8] = V_mat;

//     return rst_mat;
// }

void Save_mat(MatrixXmat mat, string file_name)
{
    ofstream outfile(file_name.c_str());
    if(! outfile)
    {
        cout<<"Error in Opening: "<<file_name<<endl;
        return;
    }

    int row = mat.rows();
    int col = mat.cols();

    for (int i = 0; i != row; i++)
    {
        for (int j= 0; j != col; j++)
        {
            outfile<<mat(i,j)<<"\t";
        }
        outfile<<endl;
    }

    outfile.close();
    outfile.clear();
}

MatrixXmat Load_mat(string file_name, int row, int col)
{
    MatrixXmat mat(row, col);

    ifstream infile(file_name.c_str());
    if(!infile)
    {
        cout<<"Error in Opening: "<<file_name<<endl;
        exit(-1);
    }

    int i = 0;
    while(!infile.eof())
    {
        int j = 0;

        string line("");
        getline(infile, line);

        if(line.empty())
        {
            continue;
        }

        stringstream ss(line);

        int c = col;
        while(!ss.eof() && c--)
        {
            UnitType temp;
            ss>>temp;
            mat(i, j++) = temp;
        }

        i++;
    }

    infile.close();
    infile.clear();

    return mat;
}

void Save_mat(vector<vector<MatrixXmat > > mat, string file_name)
{
    ofstream outfile(file_name.c_str());
    if(! outfile)
    {
        cout<<"Error in Opening: "<<file_name<<endl;
        return;
    }

    if(mat.empty())
    {
        cout<<"mat is empty!"<<endl;
        return;
    }

    int vecRow = mat.size();
    int vecCol = mat[0].size();

    int row = mat[0][0].rows();
    int col = mat[0][0].cols();

    for (int i = 0; i != vecRow; i++){
        for (int j= 0; j != vecCol; j++){
            for (int k = 0; k != row; k++){
                for (int h = 0; h != col; h++){
                    outfile<<mat[i][j](k, h)<<"\t";
                }
                outfile<<endl;
            }
            outfile<<endl;
        }

    }

    outfile.close();
    outfile.clear();
}

vector<vector<MatrixXmat > > Load_mat(string file_name, int vecRow, int vecCol, int row, int col){
    vector<vector<MatrixXmat > > mat;

    ifstream infile(file_name.c_str());
    if(!infile){
        cout<<"Error in Opening: "<<file_name<<endl;
        exit(-1);
    }

    int i = 0;
    vector<MatrixXmat >  tempMat;
    MatrixXmat  tempTempMat(row, col);

    while(!infile.eof()){
        int j = 0;

        string line("");
        getline(infile, line);

        if(line.empty())
        {
            continue;
        }

        stringstream ss(line);

        int c = col;
        while(!ss.eof() && c--){
            UnitType temp;
            ss>>temp;
            tempTempMat(i, j++) = temp;
        }

        if(i == row - 1){
            tempMat.push_back(tempTempMat);
            i = -1;
            if(tempMat.size() == vecCol)
            {
                mat.push_back(tempMat);
                tempMat.clear();
            }
        }
        i++;
    }

    infile.close();
    infile.clear();

    return mat;
}

