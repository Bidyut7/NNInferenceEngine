//
//  Matrix.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#include "Matrix.hpp"
#include <stdexcept> //for throwing exceptions
#include <Accelerate/Accelerate.h>


//Constructor Implementation
Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f){
    //initialize data with r*c elements, all set to 0.0f
    if (r<=0 || c<=0){
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
}

//Destructor Implementation
Matrix::~Matrix(){
    
}

//get_implementation
float Matrix::get_value(int r, int c) const{
    if (r<0 || r>=rows || c<0 || c>=cols){
        throw std::out_of_range("matrix indices out of bound");
    }
    return data[r * cols + c]; //Row-major order
}

//set_implementation
void Matrix::set_value(int r, int c, float val){
    if (r<0 || r>=rows || c<0 || c>=cols){
        throw std::out_of_range("Matrix indices out of bound");
    }
    data[r * cols + c] = val; //Row-major order
}

//printing Implementation
void Matrix::print() const{
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            std::cout << data[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

Matrix add(const Matrix& A, const Matrix& B){
    if (A.rows!=B.rows || A.cols!=B.cols){
        throw std::invalid_argument("Matrix dimension do not match");
    }
    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] + B.data[i];
    }
    return result;
}

Matrix subtract(const Matrix& A, const Matrix& B){
    if (A.rows != B.rows || A.cols != B.cols){
        throw std::invalid_argument("Matrix doesn't have equal dimensions");
    }
    Matrix result(A.rows, A.cols);
    for (int i=0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] - B.data[i];
    }
    return result;
}

Matrix multiply(const Matrix& A, const Matrix& B){
    if (A.rows!=B.rows || A.cols!=B.cols){
        throw std::invalid_argument("matrix dimesnions do not match");
    }
    Matrix result(A.rows, A.cols);
    for (int i=0; i < A.rows; ++i){
        for (int j = 0; j < B.cols; ++j){
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k){
                sum += A.get_value(i, k) * B.get_value(k, j);
            }
            result.set_value(i, j, sum);
        }
    }
    return result;
}

Matrix multiply_accelerate(const Matrix& A, const Matrix& B){
    if (A.cols != B.rows){
        throw std::invalid_argument("matrix A column must match matrix B row");
    }
    Matrix C(A.rows, A.cols);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0f, A.data.data(), A.cols, B.data.data(), B.cols, 0.0f, C.data.data(), C.cols);
    return C;
}
