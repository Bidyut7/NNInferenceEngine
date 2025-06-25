//
//  QuantizationUtils.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 18/06/25.
//

#include "QuantizationUtils.hpp"
#include <limits> //for std::numeric_limits
#include <iostream>
#include <cmath>
#include <iomanip>

namespace QuantizationUtils {

// calculate_quant_params is now simpler, it only determines the scale
void calculate_quant_params(float min_float, float max_float,
                            int8_t min_int_target, int8_t max_int_target, float& scale) { // zero_point removed as an output

    std::cout << "\nDEBUG - Entering calculate_quant_params:\n";
    std::cout << "DEBUG -   min_float: " << std::fixed << std::setprecision(8) << min_float << "\n";
    std::cout << "DEBUG -   max_float: " << std::fixed << std::setprecision(8) << max_float << "\n";
    std::cout << "DEBUG -   min_int_target (param): " << (int)min_int_target << "\n";
    std::cout << "DEBUG -   max_int_target (param): " << (int)max_int_target << "\n";

    // Handle edge case where the float range is zero (all values are the same)
    if (std::fabs(max_float - min_float) < std::numeric_limits<float>::epsilon()) {
        scale = 1.0f; // Arbitrary non-zero scale to prevent division by zero
        std::cerr << "WARNING: Float range is zero. Using default scale=1.0.\n";
        std::cout << "DEBUG - Exiting calculate_quant_params (flat range).\n\n";
        return;
    }

    // Scale calculation remains the same: map full float range to full integer range
    scale = (max_float - min_float) / (static_cast<float>(max_int_target) - static_cast<float>(min_int_target));
    std::cout << "DEBUG - Calculated scale: " << std::fixed << std::setprecision(8) << scale << "\n";
    std::cout << "DEBUG - Exiting calculate_quant_params.\n\n";
}

// quantize_float_to_int8 now takes min/max float and int targets directly
int8_t quantize_float_to_int8(float val,
                              float min_float, float max_float,
                              int8_t min_int_target, int8_t max_int_target) {
    // Handle edge case for flat range (should ideally be handled by calculate_quant_params, but safety check)
    if (std::fabs(max_float - min_float) < std::numeric_limits<float>::epsilon()) {
        return (min_int_target + max_int_target) / 2; // Map to middle of integer range
    }

    // Linearly map the float value from its range to the integer target range
    // val_normalized = (val - min_float) / (max_float - min_float)  (maps val to 0-1)
    // quantized_val_float = min_int_target + val_normalized * (max_int_target - min_int_target)
    float val_normalized = (val - min_float) / (max_float - min_float);
    float quantized_val_float = static_cast<float>(min_int_target) + val_normalized * (static_cast<float>(max_int_target) - static_cast<float>(min_int_target));

    // Round to the nearest integer
    int32_t quantized_val_int32 = static_cast<int32_t>(std::round(quantized_val_float));

    // Explicitly clamp the quantized value to the int8_t range [-128, 127]
    if (quantized_val_int32 < -128) return -128;
    if (quantized_val_int32 > 127) return 127;

    return static_cast<int8_t>(quantized_val_int32);
}

// dequantize_int8_to_float now takes min/max float and int targets directly
float dequantize_int8_to_float(int8_t val,
                               float min_float, float max_float,
                               int8_t min_int_target, int8_t max_int_target) {
    // Handle edge case for flat range
    if (std::fabs(max_float - min_float) < std::numeric_limits<float>::epsilon()) {
        return min_float; // Return the single float value
    }

    // Linearly map the integer value from its range back to the float range
    // val_normalized = (val - min_int_target) / (max_int_target - min_int_target) (maps val to 0-1)
    // dequantized_val_float = min_float + val_normalized * (max_float - min_float)
    float val_normalized = (static_cast<float>(val) - static_cast<float>(min_int_target)) / (static_cast<float>(max_int_target) - static_cast<float>(min_int_target));
    float dequantized_val_float = min_float + val_normalized * (max_float - min_float);

    return dequantized_val_float;
}

}
