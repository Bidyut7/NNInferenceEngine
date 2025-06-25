//
//  QuantizationUtils.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 18/06/25.
//

#ifndef QuantizationUtils_hpp
#define QuantizationUtils_hpp

#include <stdio.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace QuantizationUtils { //using namespace to keep the functions organized
//helper to calculate scale and zero_point for a given float range and int range
void calculate_quant_params(float min_float, float max_float, int8_t min_int, int8_t max_int, float& scale);

//quantize a float value to int
int8_t quantize_float_to_int8(float val,
                              float min_float, float max_float,
                              int8_t min_int_target, int8_t max_int_target);

//dequantize an int8 value to float
float dequantize_int8_to_float(int8_t val,
                               float min_float, float max_float,
                               int8_t min_int_target, int8_t max_int_target);
}

#endif /* QuantizationUtils_hpp */
