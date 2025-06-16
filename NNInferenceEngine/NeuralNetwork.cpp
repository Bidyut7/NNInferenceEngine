//
//  NeuralNetwork.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 16/06/25.
//

#include "NeuralNetwork.hpp"
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "ActivationLayer.hpp"
#include "DenseLayer.hpp"
#include "Layer.hpp"

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer){
    layers.push_back(std::move(layer)); //using std::move to transfer ownership
}

Matrix NeuralNetwork::predict(const Matrix& input){
    Matrix current_output = input;
    
    for (const auto& layer : layers){
        current_output = layer -> forward(current_output); //passing through each layer
    }
    return current_output;
}

Matrix NeuralNetwork::load_matrix_from_csv(const std::string& file_path, int expected_rows, int expected_cols){
    std::ifstream file(file_path);
    if (!file.is_open()){
        throw std::runtime_error("Could not open CSV: " + file_path);
    }
        std::vector<float> data_values;
        std::string line;
        int current_rows = 0;
        int current_cols = 0;
        bool first_line = true;
        
        while (std::getline(file, line)){
        std::stringstream ss(line);
        std::string cell;
        int count_cols_in_line = 0;
        while (std::getline(ss, cell, ',')){
            data_values.push_back(std::stof(cell));
            count_cols_in_line++;
        }
        if (first_line){
            current_cols = count_cols_in_line;
            first_line = false;
        } else if (current_cols != count_cols_in_line){
            throw std::runtime_error("CSV file has inconsistent column counts: " + file_path);
        }
        current_rows++;
    }
        if (current_rows != expected_rows || current_cols != expected_cols) {
         throw std::runtime_error("CSV file dimensions do not match expected dimensions for " + file_path + ". Expected: " + std::to_string(expected_rows) + "x" + std::to_string(expected_cols) +    ", Got: " + std::to_string(current_rows) + "x" + std::to_string(current_cols));
    }
        Matrix loaded_matrix(current_rows, current_cols);
        for (size_t i = 0; i < data_values.size(); ++i) {
            loaded_matrix.data[i] = data_values[i];
    }
    return loaded_matrix;
}

void NeuralNetwork::load_model_from_csv(const std::string& folder_path){
    //clear existing layer
    layers.clear();
    
    //for this example we're assuming a 2 layer neural network
    //input -> Dense1 (2 inputs, 3 outputs) -> ReLU -> Dense2 (3 inputs, 1 output) -> Sigmoid
    // Layer 1: Dense Layer
    // Weights: 2x3, Biases: 1x3
    std::string dense1_weights_path = folder_path + "/dense1_weights.csv";
    std::string dense1_biases_path = folder_path + "/dense1_biases.csv";
    
    std::cout<<"Loading Dense Layer 1 weights from: " << dense1_weights_path << std::endl;
    Matrix dense1_weights = load_matrix_from_csv(dense1_weights_path, 2, 3);
    std::cout << "Loading Dense Layer 1 biases from: " << dense1_biases_path << std::endl;
    Matrix dense1_biases = load_matrix_from_csv(dense1_biases_path, 1, 3);
    
    auto dense1_layer = std::make_unique<DenseLayer>(2,3);
    dense1_layer->weights = dense1_weights;
    dense1_layer->biases = dense1_biases;
    add_layer(std::move(dense1_layer));
    add_layer(std::make_unique<ActivationLayer>(ActivationType::ReLU));
    
    // Layer 2: Dense Layer
    // Weights: 3x1, Biases: 1x1
    std::string dense2_weights_path = folder_path + "/dense2_weights.csv";
    std::string dense2_biases_path = folder_path + "/dense2_biases.csv";
    
    std::cout<<"Loading Dense Layer 2 weights from: " << dense2_weights_path << std::endl;
    Matrix dense2_weights = load_matrix_from_csv(dense2_weights_path, 3, 1);
    std::cout << "Loading Dense Layer 2 biases from: " << dense2_biases_path << std::endl;
    Matrix dense2_biases = load_matrix_from_csv(dense2_biases_path , 1, 1);
    
    auto dense2_layer = std::make_unique<DenseLayer>(3, 1);
    dense2_layer->weights = dense2_weights;
    dense2_layer->biases = dense2_biases;
    add_layer(std::move(dense2_layer));
    add_layer(std::make_unique<ActivationLayer>(ActivationType::Sigmoid));
    
    std::cout << "Model loaded with " << layers.size() << " layers." << std::endl;
}
