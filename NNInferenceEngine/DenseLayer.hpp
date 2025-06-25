//
//  DenseLayer.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#ifndef DenseLayer_hpp
#define DenseLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Matrix.hpp"
#include <vector>
#include <cstdint>

class DenseLayer: public Layer{
public:
//    Matrix weights;
//    Matrix biases;
    Matrix float_weights;
    Matrix float_biases;
    
    std::vector<int8_t> quantized_weights_data;
    float weights_scale;
    float weights_min_float;
    float weights_max_float;

//    int8_t weights_zero_point;
    
    DenseLayer(int num_inputs, int num_outputs);
    
    Matrix forward(const Matrix& input) override;
    
    void quantize_weights();

private:
    void calculate_input_quant_params(const Matrix& input_val, float& min_i, float& max_i, float& scale_i);
};

#endif 
