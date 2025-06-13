//
//  VectorOps.h
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 12/06/25.
//

#ifndef VectorOps_h
#define VectorOps_h

#include <vector>

void vector_add_scalar(const float* a, const float* b, float* result, int length);

//NEON SIMD vector addition
void vector_add_neon(const float* a, const float* b, float* result, int length);


#endif /* VectorOps_h */
