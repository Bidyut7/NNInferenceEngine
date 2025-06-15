//
//  DenseLayer.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#ifndef DenseLayer_hpp
#define DenseLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Matrix.hpp"

class DenseLayer: public Layer{
public:
    Matrix weights;
    Matrix baises;
    
    DenseLayer(int num_inputs, int num_outputs);
    
    Matrix forward(const Matrix& input) override;
};

#endif /* DenseLayer_hpp */
