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

