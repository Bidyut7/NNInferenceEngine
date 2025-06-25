//
//  InferencePipeline.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 17/06/25.
//

#ifndef InferencePipeline_hpp
#define InferencePipeline_hpp

#include <stdio.h>
#include <memory>
#include <string>
#include "NeuralNetwork.hpp"

class InferencePipeline{
public:
    std::unique_ptr<NeuralNetwork> network;
    
    InferencePipeline();
    
    //loading the neural network model
    void load_model(const std::string& model_folder_path);
    
    //running inference on a given input and returing a output
    Matrix run_inference(const Matrix& input, bool print_layer_timings = false);
    
    Matrix preprocess_input(const Matrix& raw_input);
    Matrix postprocess_output(const Matrix& raw_output);
};

#endif 
