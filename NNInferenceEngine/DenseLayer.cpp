//
//  DenseLayer.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#include "DenseLayer.hpp"
#include "Matrix.hpp"
#include <stdexcept>
#include <iostream>
#include "QuantizationUtils.hpp"
#include <iomanip>
#include <vector>
#include "QuantizedTensor.h"
#include <Accelerate/Accelerate.h>


DenseLayer::DenseLayer(int num_inputs, int num_outputs)
    : float_weights(num_inputs, num_outputs),
      float_biases(1, num_outputs),
      weights_scale(1.0f),
      weights_min_float(0.0f),
      weights_max_float(0.0f)
{
    // Default initialization of internal data, will be overwritten by loaded model
}

// Private helper for input quantization params (stub implementation)
void DenseLayer::calculate_input_quant_params(const Matrix& input_val, float& min_i, float& max_i, float& scale_i) {
    min_i = 0.0f;
    max_i = 1.0f;
    scale_i = 1.0f;
}


void DenseLayer::quantize_weights() {
    float_weights.get_min_max(weights_min_float, weights_max_float);

    QuantizationUtils::calculate_quant_params(weights_min_float, weights_max_float,
                                             (int8_t)-128, (int8_t)127, weights_scale);

    quantized_weights_data.resize(float_weights.rows * float_weights.cols);
    // std::cout << "DEBUG - Quantizing weights. Original Float Values:\n";
    // float_weights.print();

    for (size_t i = 0; i < float_weights.data.size(); ++i) {
        quantized_weights_data[i] = QuantizationUtils::quantize_float_to_int8(
            float_weights.data[i], weights_min_float, weights_max_float, (int8_t)-128, (int8_t)127);
    }
    std::cout << "DenseLayer: Weights quantized. Min/Max Float: [" << weights_min_float << ", " << weights_max_float
              << "], Scale: " << std::fixed << std::setprecision(8) << weights_scale << std::endl;

    // DEBUG - This print is fine for initial quantization, but not in forward loop
    // std::cout << "DEBUG - Quantized (int8) weights data:\n";
    // for(size_t i = 0; i < quantized_weights_data.size(); ++i) {
    //     std::cout << std::setw(4) << (int)quantized_weights_data[i] << ((i + 1) % float_weights.cols == 0 ? "\n" : "\t");
    // }
}


Matrix DenseLayer::forward(const Matrix& input) {
    if (input.cols != float_weights.rows) {
        throw std::invalid_argument("Input columns must match dense layer's input features (weights.rows).");
    }

    // std::cout << "\nDEBUG - Entering DenseLayer::forward (Layer: " << this << ")\n";

    // Create a temporary Matrix to hold dequantized weights for multiplication
    Matrix dequantized_weights(float_weights.rows, float_weights.cols);
    // std::cout << "DEBUG - Dequantizing int8_t weights back to float and comparing to original floats:\n";
    // std::cout << "Original\tDequantized\tDifference\n";
    for (int r = 0; r < dequantized_weights.rows; ++r) {
        for (int c = 0; c < dequantized_weights.cols; ++c) {
            size_t idx = r * dequantized_weights.cols + c;
            int8_t q_val = quantized_weights_data[idx];
            float deq_val = QuantizationUtils::dequantize_int8_to_float(
                q_val, weights_min_float, weights_max_float, (int8_t)-128, (int8_t)127);
            // float original_val = float_weights.data[idx]; // Only needed for debug print
            dequantized_weights.set_value(r, c, deq_val);

            // std::cout << std::fixed << std::setprecision(6) // Only for debug
            //           << original_val << "\t\t"
            //           << deq_val << "\t\t"
            //           << std::fabs(original_val - deq_val) << "\n";
        }
    }
    // std::cout << "DEBUG - Full dequantized weights matrix:\n"; // Only for debug
    // dequantized_weights.print();

    // Perform matrix multiplication: input * dequantized_weights
    // This calls multiply_accelerate, which uses cblas_sgemm
    Matrix output = multiply_accelerate(input, dequantized_weights);

    // Add biases
    for (int r = 0; r < output.rows; ++r) {
        for (int c = 0; c < output.cols; ++c) {
            output.set_value(r, c, output.get_value(r, c) + float_biases.get_value(0, c));
        }
    }

    // std::cout << "DEBUG - Output of DenseLayer::forward (before Activation):\n"; // Only for debug
    // output.print();
    // std::cout << "DEBUG - Exiting DenseLayer::forward\n"; // Only for debug
    return output;
}
