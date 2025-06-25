//
//  ActivationLayer.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#include "ActivationLayer.hpp"
#include <cmath>
#include <arm_neon.h>
#include <iostream>

float ActivationLayer::relu(float x){
    return std::max(0.0f, x);
}

float ActivationLayer::sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

//constructor
ActivationLayer::ActivationLayer(ActivationType t) : type(t) {}

// Forward pass for activation layer - NEON OPTIMIZED (without internal prints)
Matrix ActivationLayer::forward(const Matrix& input) {
    Matrix output(input.rows, input.cols);
    size_t total_elements = input.rows * input.cols;

    const float* input_data_ptr = input.data.data();
    float* output_data_ptr = output.data.data();

    size_t i = 0;
    for (; i + 3 < total_elements; i += 4) {
        float32x4_t input_vec = vld1q_f32(input_data_ptr + i);
        float32x4_t output_vec;

        switch (type) {
            case ActivationType::ReLU: {
                output_vec = vmaxq_f32(input_vec, vdupq_n_f32(0.0f));
                break;
            }
            case ActivationType::Sigmoid: {
                float temp_out[4];
                temp_out[0] = sigmoid(vgetq_lane_f32(input_vec, 0));
                temp_out[1] = sigmoid(vgetq_lane_f32(input_vec, 1));
                temp_out[2] = sigmoid(vgetq_lane_f32(input_vec, 2));
                temp_out[3] = sigmoid(vgetq_lane_f32(input_vec, 3));
                output_vec = vld1q_f32(temp_out);
                break;
            }
            default:
                throw std::runtime_error("Unknown activation type.");
        }
        vst1q_f32(output_data_ptr + i, output_vec);
    }

    for (; i < total_elements; ++i) {
        float val = input_data_ptr[i];
        float activated_val;
        switch (type) {
            case ActivationType::ReLU:
                activated_val = relu(val);
                break;
            case ActivationType::Sigmoid:
                activated_val = sigmoid(val);
                break;
            default:
                throw std::runtime_error("Unknown activation type.");
        }
        output_data_ptr[i] = activated_val;
    }
    return output;
}
