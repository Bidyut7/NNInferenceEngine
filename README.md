# NNInferenceEngine
NNInferenceEngine: A High-Performance C++ Neural Network Inference Engine
Overview
This repository hosts a high-performance neural network inference engine implemented from scratch in C++. The primary goal of this project was to understand the underlying mechanics of neural network forward passes and to optimize their execution for speed, particularly on Apple Silicon (ARM64) architectures, leveraging low-level techniques like ARM NEON intrinsics and Apple's Accelerate framework.

Unlike high-level machine learning frameworks, this project focuses on the bare-metal implementation of inference, providing insights into data structures, computational kernels, and performance bottlenecks often abstracted away.

Features
Feedforward Neural Network Support: Implements a sequential model structure.
Layer Types:
Dense (Fully Connected) Layers: Standard linear transformations.
Activation Layers: ReLU and Sigmoid activation functions.
Model Loading: Loads network weights and biases from CSV files, allowing for external model definition.
Quantization (Post-Training): Supports float to int8 quantization for weights, enabling memory efficiency and preparation for integer-only arithmetic.
High-Performance Matrix Multiplication: Integrates Apple's Accelerate framework's cblas_sgemm (BLAS) for highly optimized floating-point General Matrix Multiply (GEMM) operations.
ARM NEON Vectorization: Hand-optimized activation functions using ARM NEON intrinsics for significant speedup on ARM64 CPUs.
Performance Benchmarking: Includes std::chrono-based timing to accurately measure overall inference time and per-layer execution times.
Command-Line Interface (CLI): Easy execution with configurable model paths, input files, and number of inference runs.
Architecture
The engine is built around a modular and extensible architecture:

Matrix.hpp/Matrix.cpp:
A fundamental class for 2D float arrays, serving as the primary data structure for inputs, weights, biases, and intermediate activations.
Includes basic matrix operations and utility functions (e.g., get_min_max).
multiply_accelerate: Leverages cblas_sgemm for fast float matrix multiplication.
multiply_neon: A custom NEON-optimized float matrix multiplication (primarily for benchmarking and learning, not used in final DenseLayer for performance due to cblas_sgemm's superiority).
Layer.hpp: An abstract base class defining the forward interface, enabling polymorphism for different layer types.
DenseLayer.hpp/DenseLayer.cpp:
Implements a fully connected layer.
Stores float weights and biases.
Includes quantized_weights_data (a std::vector<int8_t>) for memory-efficient weight storage.
The forward pass dequantizes int8 weights back to float and then uses multiply_accelerate for computation.
Handles weight quantization (quantize_weights()) during model loading.
ActivationLayer.hpp/ActivationLayer.cpp:
Implements element-wise activation functions (ReLU, Sigmoid).
The forward pass is heavily optimized using ARM NEON intrinsics for faster execution.
NeuralNetwork.hpp/NeuralNetwork.cpp:
Manages the sequence of layers in the network (std::vector<std::unique_ptr<Layer>>).
Provides add_layer and predict methods.
load_model_from_csv: Handles loading weights and biases for each layer from CSV files.
Includes a static helper load_matrix_from_csv for reading CSV data into Matrix objects.
InferencePipeline.hpp/InferencePipeline.cpp:
A high-level class that orchestrates the entire inference process.
Manages loading the full model (load_model) and executing the forward pass (run_inference).
QuantizationUtils.hpp/QuantizationUtils.cpp:
Provides utility functions for float to int8 quantization and int8 to float dequantization, including robust scale and zero_point calculation.
Optimization Strategies & Performance
The core of this project's optimization lies in leveraging platform-specific hardware capabilities for common neural network operations.

Accelerate Framework (BLAS)
Strategy: For the computationally intensive matrix multiplication within DenseLayer, we utilize cblas_sgemm from Apple's Accelerate framework. This is a highly optimized Basic Linear Algebra Subprograms (BLAS) routine.
Impact: cblas_sgemm is often hand-tuned in assembly by Apple engineers, making it exceptionally fast by exploiting CPU features like SIMD (Single Instruction Multiple Data) and multi-threading implicitly. It proved to be the primary workhorse for the DenseLayer's performance.
ARM NEON Vectorization
Strategy: For element-wise operations like activation functions (ReLU, Sigmoid), which typically involve looping through every element, we implemented ARM NEON intrinsics. NEON allows the CPU to perform the same operation on multiple data points simultaneously (e.g., 4 float values at once).
Impact: This significantly reduced the execution time of ActivationLayer::forward to microseconds, making it a negligible bottleneck compared to matrix multiplication.
Quantization
Strategy: Post-training quantization from float to int8 (8-bit integers) was applied to the network's weights. This involves calculating a scale and zero_point to map the original float range to the int8 range ([-128, 127]).
Impact:
Memory Efficiency: int8 weights reduce the model's memory footprint by 4x compared to float32.
Performance (Current): In this implementation, the int8 weights are dequantized back to float just before cblas_sgemm is called. While this adds a small overhead, it allows leveraging the highly optimized cblas_sgemm. True int8 compute (performing multiplication on int8 directly) is more complex and discussed in "Future Work."
Performance Metrics
After optimizing the dense layers with Accelerate and activation layers with NEON, the engine achieved the following performance on a typical Apple Silicon Mac (e.g., M1/M2/M3):

Network Size: 3 layers (128 input features -> 256 hidden neurons -> 128 hidden neurons -> 10 output classes).
Average Time per Inference: 1.845 milliseconds
This demonstrates the effectiveness of combining a modular C++ design with platform-specific low-level optimizations for neural network inference.

Challenges Faced & Solutions
Building this high-performance engine involved navigating several complex challenges:

Initial calculate_quant_params Errors (Incorrect Scale/ZeroPoint)
Problem: Early implementations of calculate_quant_params resulted in drastically incorrect scale values (128.004 instead of 0.00196) or zero_point values (77 instead of -128 for a positive range). This led to all quantized weights saturating to the maximum int8 value (127), causing severe accuracy degradation.
Mitigation: We performed a deep dive into the mathematical formulas for asymmetric quantization. The scale formula (max_float−min_float)/(max_int−min_int) was precisely defined. For zero_point, a robust linear mapping strategy was adopted within QuantizationUtils::quantize_float_to_int8 and dequantize_int8_to_float, ensuring min_float accurately mapped to min_int_target (e.g., -128) and max_float to max_int_target (e.g., 127). Explicit int32_t intermediate types were used in rounding and clamping to prevent int8_t overflow issues during calculation.
Xcode Instruments Not Showing Application Hotspots
Problem: When profiling with Xcode Instruments' Time Profiler, the tool consistently showed only system-level dyld4 (dynamic linker) calls as the main bottleneck, even for millions of inference runs. The actual application code's execution time was too short or too aggressively optimized by the compiler to be visible.
Mitigation:
We rigorously ensured all old benchmark code was removed/commented out from main.cpp.
The number of inference runs was dramatically increased (from 10,000 to 100,000 to 1 million, and even 10 million attempts) to provide more data points for the profiler.
Crucially, a small, random variation was introduced into the input data within the main inference loop. This forced the compiler to perform the full neural network computation for each iteration, preventing it from optimizing the entire loop away as redundant.
Ultimately, we relied on direct std::chrono timers within the C++ code (specifically in NeuralNetwork::predict for per-layer times and main.cpp for total time) to get accurate, in-app performance metrics, bypassing the external profiler's limitations.
cblas_gemm_s8s8s32 Undeclared/BNNS Deprecation
Problem: An attempt to use cblas_gemm_s8s8s32 (an int8-specific BLAS function) resulted in a "file not found" error for its header or an "undeclared identifier" error. Further research indicated that Apple's BNNS (Basic Neural Network Subroutines) framework, which would be the higher-level API for int8 on CPU, is deprecated.
Mitigation: We pivoted from trying to use a potentially internal/deprecated int8 BLAS function. Instead, the strategy shifted to using the widely supported and highly optimized cblas_sgemm (float BLAS) for the core matrix multiplication. This allowed us to maintain high performance with a robust, documented API, while still leveraging int8 for memory reduction (by converting int8 weights back to float just before computation). The focus remained on general-purpose CPU optimization without delving into complex and rapidly evolving int8-specific APIs or GPU programming (like MPS Graph), which were beyond the project's initial scope.
File Not Found / CLI Argument Issues
Problem: Initial attempts to run the compiled executable from the terminal failed due to the model's CSV files not being found or incorrect command-line argument parsing. This stemmed from the models_large directory not being in the executable's runtime path and missing arguments.
Mitigation: Explicit instructions were provided to copy the models_large directory (generated by the Python script) directly into the Xcode build's DerivedData/<project-hash>/Build/Products/Debug/ directory. The main.cpp's CLI parsing logic was refined to correctly check for argc == 4 and provide clear usage instructions, guiding the user to provide all three required arguments (model_folder_path, input_file_path, num_inference_runs).
Verbose Debug Prints Slowing Benchmarking
Problem: Numerous std::cout statements inside the DenseLayer::forward and ActivationLayer::forward methods, intended for debugging, significantly slowed down the performance benchmarking loop, leading to perceived hangs or truncated output.
Mitigation: All verbose DEBUG std::cout statements within Layer::forward implementations were commented out or removed. A print_layer_timings boolean flag was added to NeuralNetwork::predict and InferencePipeline::run_inference. This allowed printing detailed layer-by-layer timings for a single "initial inference" run, while suppressing all output during the main high-iteration performance benchmarking loop, providing clean and accurate total time measurements.
Future Work
This project provides a strong foundation for building a custom inference engine. Here are several exciting areas for future development:

Full int8 Inference Pipeline:
Implement true int8 GEMM using ARM NEON intrinsics (e.g., vdotq_s32 for ARMv8.2+ or vmlal_s16 for older versions) for multiplication of int8 inputs and int8 weights, accumulating to int32.
Develop int8-aware activation functions and "re-quantization" layers to convert intermediate int32 results back to int8 for subsequent layers, enabling an end-to-end int8 data path.
More Layer Types:
Convolutional Layers: Essential for Computer Vision tasks.
Pooling Layers: Max-pooling, Average-pooling for dimensionality reduction.
Batch Normalization: Improve training stability (can be fused into dense/conv layers for inference).
Other Activation Functions: Implement Softmax (for classification outputs), Tanh, Leaky ReLU, etc.
Model Serialization/Deserialization:
Develop a custom binary format to save and load the entire neural network structure, weights, biases, and quantization parameters. This is much faster and more robust than CSVs.
Training (Backpropagation):
Extend the engine to support the training phase, including implementing the backward pass for each layer, loss functions (e.g., Mean Squared Error, Cross-Entropy), and optimizers (e.g., Stochastic Gradient Descent, Adam).
Batch Inference:
Explicitly optimize for processing multiple input samples in parallel (batching) to maximize throughput, further leveraging the parallel capabilities of cblas_sgemm.
Cross-Platform Portability:
Abstract platform-specific optimizations (Accelerate, NEON) using cross-platform libraries (e.g., Eigen, OpenBLAS, or custom SIMD with highway library) to run the engine on Windows, Linux, and other CPU architectures.
GPU Acceleration:
Integrate Apple's Metal framework (specifically Metal Performance Shaders - MPS Graph) to offload computationally intensive operations to the GPU for even higher performance.
Getting Started
Follow these steps to build and run the NNInferenceEngine on your macOS system.

Prerequisites
macOS: An Apple computer running macOS.
Xcode: Apple's integrated development environment (IDE). Install from the App Store.
Xcode Command Line Tools: Open Terminal and run xcode-select --install.
C++ Compiler: Clang (comes with Xcode).
Python 3: To generate dummy model CSVs.
1. Clone the Repository
Bash

git clone https://github.com/your-username/NNInferenceEngine.git # Replace with your repo URL
cd NNInferenceEngine
2. Generate Model CSVs
This project uses CSV files to load model weights and biases. We'll use a Python script to generate a dummy, larger model for benchmarking.

Create a Python file named generate_model_csvs.py in your project's root directory:

Python

# generate_model_csvs.py
import numpy as np
import os

def generate_csv_model(folder_path="models", layer_dims=[(2,3), (3,1)]):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    current_inputs = None
    for i, (num_inputs, num_outputs) in enumerate(layer_dims):
        if current_inputs is None:
            current_inputs = num_inputs
        if current_inputs != num_inputs and i > 0:
            print(f"Warning: Layer {i} input dim {num_inputs} does not match previous layer output {current_inputs}. Adjusting.")
            num_inputs = current_inputs

        weights = np.random.rand(num_inputs, num_outputs).astype(np.float32) * 0.1 - 0.05
        biases = np.random.rand(1, num_outputs).astype(np.float32) * 0.01 - 0.005

        weights_filename = os.path.join(folder_path, f"dense{i+1}_weights.csv")
        biases_filename = os.path.join(folder_path, f"dense{i+1}_biases.csv")

        np.savetxt(weights_filename, weights, delimiter=',', fmt='%.8f')
        np.savetxt(biases_filename, biases, delimiter=',', fmt='%.8f')

        print(f"Generated dense{i+1}_weights.csv ({num_inputs}x{num_outputs}) and biases.")
        current_inputs = num_outputs

if __name__ == "__main__":
    generate_csv_model("models_large", layer_dims=[(128, 256), (256, 128), (128, 10)])
    print("\nGenerated large model in 'models_large' folder.")

    input_data = np.random.rand(1, 128).astype(np.float32) * 0.1
    np.savetxt(os.path.join("models_large", "input.csv"), input_data, delimiter=',', fmt='%.8f')
    print("Generated dummy input.csv for large model.")
Run the script from your terminal:

Bash

python generate_model_csvs.py
This will create a models_large folder containing the necessary CSVs.

3. Build the Project in Xcode
Open the NNInferenceEngine.xcodeproj file in Xcode.
Go to Product > Clean Build Folder (hold Option key to see it).
Go to Product > Build.
4. Copy Model to Build Folder
The executable runs from a specific build directory. You need to copy the models_large folder there.

In Xcode's Project Navigator, navigate to the "Products" group (usually at the bottom of the file tree).
Right-click on NNInferenceEngine (the executable in black text).
Select "Show in Finder". This will open the specific Debug directory where your executable is located (e.g., DerivedData/.../Build/Products/Debug/).
Copy the models_large folder (generated in Step 2) into this Debug directory.
5. Run from Terminal
Navigate to the Debug directory in your Terminal (the same one you opened in Finder in the previous step) and run the executable with arguments.

Bash

# Navigate to your Debug directory (path will vary, copy from Xcode's "Show in Finder")
cd /Users/shreysharma/Library/Developer/Xcode/DerivedData/NNInferenceEngine-YOUR_UNIQUE_HASH/Build/Products/Debug/

# Run the program
# Arguments: <model_folder_path> <input_file_path> <num_inference_runs>
./NNInferenceEngine models_large models_large/input.csv 100
(You can change 100 to 1000 or 10000 for more runs if your system handles it without hanging).

You should see output similar to:

Day 9: Final Touches - Neural Network Inference Engine

--- Model Loading ---
Loading Dense Layer 1 weights from: models_large/dense1_weights.csv
...
Model loaded with 6 layers.

--- Input Loading ---

--- Initial Inference (for detailed layer timings - prints once) ---
Layer 0 type: DenseLayer took: X.XXX ms
Layer 1 type: ActivationLayer (ReLU) took: Y.YYY ms
...
Final output sample: Z.ZZZ

--- Performance Benchmarking ---
Running inference 100 times for total time measurement...
Total inference time for 100 runs: 0.18453525 seconds
Average time per inference: 1.84535250 ms

--- Final Output Verification ---
Final output sample: A.AAA
Verification: Output printed successfully. Accuracy verification needs a reference model.

Program execution completed successfully.
How Can Others Use It With Their Model?
Your current engine is designed to be highly customizable via CSV files. Here's how others can integrate and use their own models:

Model Training (External)
Users will need to train their neural network models using standard machine learning frameworks like PyTorch, TensorFlow, Keras, or scikit-learn. The model should be a feedforward (dense) network with ReLU and/or Sigmoid activation functions, as these are currently supported.

Exporting Weights and Biases to CSV
After training, the weights and biases for each dense layer need to be extracted and saved into separate CSV files.

Crucial Format:

Weights: Should be exported as a 2D CSV file (e.g., dense1_weights.csv, dense2_weights.csv). The dimensions must be (input_features,output_features) (row-major order, matching Matrix expectations).
Biases: Should be exported as a 1
timesN CSV file (e.g., dense1_biases.csv, dense2_biases.csv), where N is the number of output features for that layer.
Naming Convention: The engine expects files named dense<LayerNumber>_weights.csv and dense<LayerNumber>_biases.csv (e.g., dense1_weights.csv, dense1_biases.csv, dense2_weights.csv, etc.).
Folder Structure: All these CSVs for a single model should be placed in a dedicated folder (e.g., my_custom_model_data).
Example Python Snippet for Export (Conceptual - using PyTorch):

Python

import torch
import torch.nn as nn
import numpy as np
import os

# Example: Define a simple model in PyTorch
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Assume 'model' is your trained PyTorch model instance
# model = SimpleMLP()
# model.load_state_dict(torch.load('your_trained_model.pth')) # Load your actual trained model

# For demonstration, let's create a dummy model and save its state
model = SimpleMLP()
# (In a real scenario, you'd load a trained model here)

output_folder = "my_custom_model_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Export weights and biases for each Dense Layer
layer_idx = 1
for name, param in model.named_parameters():
    if "weight" in name:
        # PyTorch weights are typically (output_features, input_features)
        # Your C++ Matrix expects (input_features, output_features)
        # So, transpose the weight matrix.
        weights_np = param.data.numpy().T
        np.savetxt(os.path.join(output_folder, f"dense{layer_idx}_weights.csv"), weights_np, delimiter=',', fmt='%.8f')
        print(f"Exported {name} to dense{layer_idx}_weights.csv (Shape: {weights_np.shape})")
    elif "bias" in name:
        biases_np = param.data.numpy().reshape(1, -1) # Ensure 1xN shape
        np.savetxt(os.path.join(output_folder, f"dense{layer_idx}_biases.csv"), biases_np, delimiter=',', fmt='%.8f')
        print(f"Exported {name} to dense{layer_idx}_biases.csv (Shape: {biases_np.shape})")
        layer_idx += 1 # Increment layer index after both weight and bias for a layer are processed

# Create a dummy input CSV for the custom model
custom_input_data = np.random.rand(1, 128).astype(np.float32)
np.savetxt(os.path.join(output_folder, "input.csv"), custom_input_data, delimiter=',', fmt='%.8f')
print(f"Generated dummy input.csv for custom model (Shape: {custom_input_data.shape}).")
Adjust NeuralNetwork::load_model_from_csv in C++
The NeuralNetwork::load_model_from_csv method currently has hardcoded dimensions (l1_inputs, l1_outputs, etc.) and a fixed sequence of layers (Dense -> ReLU -> Dense -> ReLU -> Dense -> Sigmoid).

Users will need to modify this method (NeuralNetwork.cpp) to accurately reflect their custom model's architecture:

Number of Layers: Add or remove blocks for DenseLayer and ActivationLayer as needed.
Layer Dimensions: Update lX_inputs and lX_outputs to match the dimensions of their exported CSVs.
Activation Functions: Change ActivationType::ReLU or ActivationType::Sigmoid to match their model's activation functions for each layer.
Prepare Input Data
Create a CSV file for their input data (e.g., my_custom_input.csv).
This file should be a single row (batch size 1) with N columns, where N is the number of input features of their model (e.g., 1
times128 for the example model).
Run with Custom Model
Compile the C++ project after modifying NeuralNetwork.cpp.

Copy their custom model folder (e.g., my_custom_model_data) and their input CSV (my_custom_input.csv) into the Debug build directory (as described in "Getting Started" Step 4).

Run the executable from Terminal, providing their custom model folder and input CSV:

Bash

./NNInferenceEngine my_custom_model_data my_custom_model_data/input.csv 100
