//
//  Convolution.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 13/06/25.
//

#ifndef Convolution_hpp
#define Convolution_hpp

#include <stdio.h>
#include "Matrix.hpp"

//Saclar 2D convolution operation
Matrix convolve_scalar(const Matrix& input, const Matrix& kernel);
Matrix convolve_neon(const Matrix& input, const Matrix& kernel);

#endif /* Convolution_hpp */
