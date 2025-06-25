//
//  QuantizedTensor.h
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 21/06/25.
//

#ifndef QuantizedTensor_h
#define QuantizedTensor_h
#include <cstdint>
#include <iostream>

struct QuantizedTensor {
    std::vector<int8_t> data;
    int rows;
    int cols;
    float min_float;
    float max_float;
    float scale;
    
    QuantizedTensor(int r, int c) : rows(r), cols(c), data(r * c), min_float(0.0f), max_float(0.0f), scale(1.0f){}
    
    // for debugging
    void print() const{
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                std::cout << (int)data[i * cols + j] << "\t";
            }
            std::cout<<std::endl;
        }
    }
};

//for output of int8 * int32
struct QuantizedAccumulatorTensor {
    std::vector<int32_t> data;
    int rows;
    int cols;
    // This would also have its own scale/zero_point for the int32 range if used in a full int8 pipeline
    // float output_scale;
    // int8_t output_zero_point;
    
    QuantizedAccumulatorTensor(int r, int c) : rows(r), cols(c), data(r * c, 0){}
    void print() const{
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                std::cout<<data[i * cols + j] << "\t";
            }
            std::cout << std::endl;
        }
    }
};

#endif 
