//
//  Matrix.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <Accelerate/Accelerate.h> // for cblas_sgemm
#include "QuantizedTensor.h"

class Matrix{
public:
    //member variable to store dimensions and data
    int rows;
    int cols;
    std::vector<float> data;
    
    //constructor
    Matrix(int r, int c);
    
    //destructor
    ~Matrix();
    
    //Accessor Methods
    float get_value(int r, int c) const;
    void set_value(int r, int c, float val);
    
    //Method to print the matrix(for debugging)
    void print() const;
    
    void get_min_max(float& min_val, float& max_val) const;
    
    struct QuantizedTensor;
    struct QuantizedAccumulatorTensor;
    
    //function for int8 multiplication
    QuantizedAccumulatorTensor multiply_quantized(const QuantizedTensor& A, const QuantizedTensor& B);

    
};

Matrix add(const Matrix& A, const Matrix& B);
Matrix subtract(const Matrix& A, const Matrix& B);
Matrix multiply(const Matrix& A, const Matrix& B);
Matrix multiply_accelerate(const Matrix& A, const Matrix& b);
Matrix multiply_neon(const Matrix& A, const Matrix& B);


#endif /* Matrix_hpp */
