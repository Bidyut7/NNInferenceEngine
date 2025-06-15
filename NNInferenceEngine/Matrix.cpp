//
//  Matrix.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "Matrix.hpp"
#include <stdexcept> //for throwing exceptions
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>


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
    Matrix result(A.rows, B.cols);
    for (int i = 0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] + B.data[i];
    }
    return result;
}

Matrix subtract(const Matrix& A, const Matrix& B){
    if (A.rows != B.rows || A.cols != B.cols){
        throw std::invalid_argument("Matrix doesn't have equal dimensions");
    }
    Matrix result(A.rows, B.cols);
    for (int i=0; i < A.rows * A.cols; ++i){
        result.data[i] = A.data[i] - B.data[i];
    }
    return result;
}

Matrix multiply(const Matrix& A, const Matrix& B){
    if (A.cols!=B.rows){
        throw std::invalid_argument("matrix dimesnions do not match");
    }
    
    Matrix result(A.rows, B.cols);
    
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
    Matrix C(A.rows, B.cols);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0f, A.data.data(), A.cols, B.data.data(), B.cols, 0.0f, C.data.data(), C.cols);
    return C;
}

Matrix multiply_neon(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix A columns must match Matrix B rows for multiplication.");
    }

    Matrix C(A.rows, B.cols);

    // Looping over rows of A (which corresponds to rows of C)
    for (int i = 0; i < A.rows; ++i) {
        // Looping over columns of B (which corresponds to columns of C)
        // Process 4 columns of C at a time using NEON vectors
        int j = 0;
        for (; j + 3 < B.cols; j += 4) {
            // Initialize 4-element accumulator for the current output C row segment
            float32x4_t c_out_vec = vdupq_n_f32(0.0f);

            // Inner loop over K dimension (columns of A / rows of B)
            int k = 0;
            for (; k + 3 < A.cols; k += 4) { // Process 4 elements along K dimension
                // Loading 4 elements from A's current row (contiguous)
                // A[i][k], A[i][k+1], A[i][k+2], A[i][k+3]
                float32x4_t a_seg = vld1q_f32(&A.data[i * A.cols + k]);

                // Loading 4x4 block from B. These loads are contiguous for rows of B.
                float32x4_t b_row0 = vld1q_f32(&B.data[k * B.cols + j]);
                float32x4_t b_row1 = vld1q_f32(&B.data[(k + 1) * B.cols + j]);
                float32x4_t b_row2 = vld1q_f32(&B.data[(k + 2) * B.cols + j]);
                float32x4_t b_row3 = vld1q_f32(&B.data[(k + 3) * B.cols + j]);

                // Performing multiply-accumulate using compile-time constants for lane indices
                c_out_vec = vmlaq_f32(c_out_vec, vdupq_n_f32(vgetq_lane_f32(a_seg, 0)), b_row0);
                c_out_vec = vmlaq_f32(c_out_vec, vdupq_n_f32(vgetq_lane_f32(a_seg, 1)), b_row1);
                c_out_vec = vmlaq_f32(c_out_vec, vdupq_n_f32(vgetq_lane_f32(a_seg, 2)), b_row2);
                c_out_vec = vmlaq_f32(c_out_vec, vdupq_n_f32(vgetq_lane_f32(a_seg, 3)), b_row3);
            }

            // After the vectorized K loop, handle any remaining K elements (tail processing)
            for (; k < A.cols; ++k) {
                // Load the scalar value from A
                float a_val = A.data[i * A.cols + k];
                // Load 4 values from B row
                float32x4_t b_vals = vld1q_f32(&B.data[k * B.cols + j]);
                // Multiply and accumulate
                c_out_vec = vmlaq_n_f32(c_out_vec, b_vals, a_val);
            }

            // Store the computed 4 output elements C[i][j...j+3] to the result matrix
            vst1q_f32(&C.data[i * C.cols + j], c_out_vec);
        } // End for j (B.cols) loop (vectorized part)

        // Scalar tail processing for the J dimension (if B.cols is not a multiple of 4)
        for (; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            C.set_value(i, j, sum);
        }
    }
    return C;
}
