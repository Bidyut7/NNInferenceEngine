//
//  DenseLayer.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#include "DenseLayer.hpp"
#include <stdexcept>

//initializing weights and baises metrices/vector
DenseLayer::DenseLayer(int num_inputs, int num_outputs): weights(num_inputs, num_outputs), baises(1, num_outputs)
{
    
}

//forward pass for a dense layer output = input*weight + bias
Matrix DenseLayer::forward(const Matrix& input){
    
    if (input.cols != weights.rows){
        throw std::invalid_argument("Input columns must match dense layer inputs(weights.rows)");
    }
    
    //performing matrix multiplication input*weights using accelerate
    Matrix output = multiply_accelerate(input, weights);
    
    //add biases to each row of the input
    for (int r = 0; r < output.rows; ++r){
        for (int c = 0; c < output.cols; ++c){
            output.set_value(r, c, output.get_value(r, c) + baises.get_value(0, c));
        }
    }
    return output;
}
