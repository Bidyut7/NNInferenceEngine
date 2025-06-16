//
//  ActivationLayer.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 15/06/25.
//

#ifndef ActivationLayer_hpp
#define ActivationLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include <functional>

//defining enums to specify activation function type
enum class ActivationType{
    ReLU,
    Sigmoid
};

class ActivationLayer: public Layer{
public:
    ActivationType type;
    
    //constructor
    ActivationLayer(ActivationType t);
    
    //overriding the forward method
    Matrix forward(const Matrix& input) override;
    
private:
    //helper function for activation
    static float relu(float x);
    static float sigmoid(float x);
};

#endif /* ActivationLayer_hpp */
