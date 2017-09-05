#ifndef MATHTYPE_H_INCLUDED
#define MATHTYPE_H_INCLUDED
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MFixedPoint/Fp32f.hpp"
#include "MFixedPoint/Fp32s.hpp"
#include "MFixedPoint/Fp64f.hpp"
#include "MFixedPoint/Fp64s.hpp"

using namespace std;
using namespace Eigen;
using namespace Fp;

//double-precision
//typedef Matrix<double, Dynamic, Dynamic> MatrixXmat;
//typedef double UnitType;

//single-precision
typedef Matrix<float, Dynamic, Dynamic> MatrixXmat;
typedef float UnitType;

//fixed-point precision
//typedef Fp32f<8> UnitType;
//typedef Matrix<UnitType, Dynamic, Dynamic> MatrixXmat;

//Sparse Matrix Defining here
typedef SparseMatrix<UnitType> MatrixSparseMat;

#endif // MATHTYPE_H_INCLUDED
