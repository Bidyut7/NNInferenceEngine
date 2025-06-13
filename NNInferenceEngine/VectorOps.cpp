//
//  VectorOps.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 12/06/25.
//

#include <stdio.h>
#include "VectorOps.h"
#include <arm_neon.h>

//scalar non SIMD vector addition
void vector_add_scalar(const float* a, const float* b, float* result, int length){
    for (int i = 0; i < length; ++i){
        result[i] = a[i] + b[i];
    }
}

//NEON SIMD vector addition
void vector_add_neon(const float* a, const float* b, float* result, int length){
    //process 4 float at a time using NEON intrinsic
    int i=0;
    // Iterate in chunks of 4 (sizeof(float32x4_t) / sizeof(float))
    for (;i + 3<length;i+=4){
        float32x4_t vec_a = vld1q_f32(a + i);
        float32x4_t vec_b = vld1q_f32(b + i);
        float32x4_t vec_sum = vaddq_f32(vec_a, vec_b);
        vst1q_f32(result + i, vec_sum);
    }
    
    // processing the last elements that don't make a full 4-element vector
    for (; i < length; ++i) {
            result[i] = a[i] + b[i];
        }
}

