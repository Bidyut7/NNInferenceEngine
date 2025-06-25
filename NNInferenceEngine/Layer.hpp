//
//  Layer.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include "Matrix.hpp"

class Layer{
public:
    
    virtual ~Layer() = default;
    
    //it represents tensors in the form of an matrix
    virtual Matrix forward(const Matrix& input) = 0;
};


#endif 
