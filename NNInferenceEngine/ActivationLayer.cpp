//
//  ActivationLayer.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#include "ActivationLayer.hpp"
#include <cmath>

float ActivationLayer::relu(float x){
    return std::max(0.0f, x);
}

float ActivationLayer::sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

//constructor
ActivationLayer::ActivationLayer(ActivationType t) : type(t) {}

//forward pass for activation layer
Matrix ActivationLayer::forward(const Matrix& input){
    Matrix output(input.rows, input.cols);
    
    for (int i = 0; i < input.rows; ++i){
        for (int j = 0; j < input.cols; ++j){
            float val = input.get_value(i, j);
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
            output.set_value(i, j, activated_val);
        }
    }
    return output;
}
