//
//  InferencePipeline.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 17/06/25.
//

#include "InferencePipeline.hpp"
#include <iostream>
#include <stdexcept>

InferencePipeline::InferencePipeline(){
    network = std::make_unique<NeuralNetwork>();
}

void InferencePipeline::load_model(const std::string& model_folder_path){
    if (!network){
        throw std::runtime_error("Neural Network not initialised in pipeline");
    }
    std::cout<<"Inference loading model from: "<<model_folder_path<<std::endl;
    network->load_model_from_csv(model_folder_path);
    std::cout<<"Inference model loaded successfully."<<std::endl;
    
}

Matrix InferencePipeline::run_inference(const Matrix& input){
    if (!network){
        throw std::runtime_error("Neural network not initialised in pipeline");
    }
    std::cout<<"Inference Pipeline: Running Inference.."<<std::endl;
    return network->predict(input);
}
