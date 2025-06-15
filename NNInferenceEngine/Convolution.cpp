//
//  Convolution.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 13/06/25.
//

#include "Convolution.hpp"
#include <stdexcept>
#include <arm_neon.h>

//Scalar 2D convolution
Matrix convolve_scalar(const Matrix& input, const Matrix& kernel){
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0){
        //for now we'll assume odd kernel dimensions
        throw std::invalid_argument("Kernel dimensions must be odd for this simple scalar convolution.");
    }
    int output_rows = input.rows - kernel.rows + 1;
    int output_cols = input.cols - kernel.cols + 1;
    
    if (output_rows <= 0 || output_cols <= 0) {
        throw std::invalid_argument("Input matrix too small for convolution with this kernel size.");
    }
    
    Matrix output(output_rows, output_cols);
    
    int kernel_centre_rows = kernel.rows / 2;
    int kernel_centre_cols = kernel.cols / 2;
    
    for (int i = 0; i < output_rows; ++i){ //input rows
        for (int j = 0; j < output_cols; ++j){ //output cols
            float sum = 0.0f;
            for (int kr = 0; kr < kernel.rows; ++kr){ // kernel rows
                for (int kc = 0; kc < kernel.cols; ++kc){ // kernel columns
                    int input_rows = i + kr; //mapping kernel to input
                    int input_col = j + kc;
                    
                    //Implementing just basic boundary check no padding
                    if (input_rows >= 0 && input_rows < input.rows && input_col >= 0 && input_col < input.cols){
                        sum += input.get_value(input_rows, input_col) * kernel.get_value(kr, kc);
                    }
                }
            }
            output.set_value(i, j, sum);
        }
    }
    return output;
}

Matrix convolve_neon(const Matrix& input, const Matrix& kernel){
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0) {
        throw std::invalid_argument("Kernel dimensions must be odd for this simple NEON convolution.");
    }
    
    int output_rows = input.rows - kernel.rows + 1;
    int output_cols = input.cols - kernel.cols + 1;
    
    if (output_rows <= 0 || output_cols <= 0) {
        throw std::invalid_argument("Input matrix too small for convolution with this kernel size.");
    }
    
    Matrix output(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; ++i){
        for (int j = 0; j < output_cols; ++j){
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            float current_output_val = 0.0f;
            
            for (int kr = 0; kr < kernel.rows; ++kr){
                int k = 0;
                for (; k + 3 < kernel.cols; k += 4){
                    int input_row = i + kr;
                    int input_col_start = j + k;
                    
                    //loading 4 contigous element from input patch
                    float input_patch_elements[4];
                    for (int lane = 0; lane < 4; ++lane){
                        if (input_row >= 0 && input_row < input.rows && input_col_start + lane >= 0 && input_col_start + lane < input.cols){
                            input_patch_elements[lane] = input.get_value(input_row, input_col_start + lane);
                        }else{
                            input_patch_elements[lane] = 0.0f;
                        }
                    }
                    float32x4_t input_seg = vld1q_f32(input_patch_elements);
                    
                    //loading 4 contigous element from kernel row
                    float32x4_t kernel_seg = vld1q_f32(&kernel.data[kr * kernel.cols + k]);
                    
                    //Multiply and accumulate
                    sum_vec = vmlaq_f32(sum_vec, input_seg, kernel_seg);
                }
                //sum the element in the accumulator for this kernel row
                current_output_val += vgetq_lane_f32(sum_vec, 0) +
                                      vgetq_lane_f32(sum_vec, 1) +
                                      vgetq_lane_f32(sum_vec, 2) +
                vgetq_lane_f32(sum_vec, 3);
                sum_vec = vdupq_n_f32(0.0f); //reset for next kernel row
                
                //Handling remaining kernel columns (tail processing)
                for (; k < kernel.cols; ++k){
                    int input_row = i + kr;
                    int input_col = j + k;
                    if (input_row >= 0 && input_row < input.rows && input_col >= 0 && input_col < input.cols) {
                    current_output_val += input.get_value(input_row, input_col) * kernel.get_value(kr, k);
                    }
                }
            }
            output.set_value(i, j, current_output_val);

        }
    }
    return output;
}
