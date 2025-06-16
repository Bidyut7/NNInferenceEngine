//
//  NeuralNetwork.hpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 16/06/25.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "Matrix.hpp"

class NeuralNetwork {
public:
    std::vector<std::unique_ptr<Layer>> layers; //storing layers using smart pointers
    
    NeuralNetwork() = default;
    
    void add_layer(std::unique_ptr<Layer> layer);
    
    Matrix predict(const Matrix& input);
    
    void load_model_from_csv(const std::string& folder_path);
    
private:
    Matrix load_matrix_from_csv(const std::string& file_path, int expected_rows, int expected_cols);
};

#endif /* NeuralNetwork_hpp */
