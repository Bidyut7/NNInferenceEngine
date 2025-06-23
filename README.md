# NNInferenceEngine: A High-Performance C++ Neural Network Inference Engine

## Overview

This repository hosts a high-performance neural network inference engine implemented from scratch in C++. The primary goal of this project was to understand the underlying mechanics of neural network forward passes and to optimize their execution for speed, particularly on Apple Silicon (ARM64) architectures, leveraging low-level techniques like ARM NEON intrinsics and Apple's Accelerate framework.

Unlike high-level machine learning frameworks, this project focuses on the bare-metal implementation of inference, providing insights into data structures, computational kernels, and performance bottlenecks often abstracted away.

## Features

- **Feedforward Neural Network Support**: Implements a sequential model structure
- **Layer Types**:
  - Dense (Fully Connected) Layers: Standard linear transformations
  - Activation Layers: ReLU and Sigmoid activation functions
- **Model Loading**: Loads network weights and biases from CSV files, allowing for external model definition
- **Quantization (Post-Training)**: Supports float to int8 quantization for weights, enabling memory efficiency and preparation for integer-only arithmetic
- **High-Performance Matrix Multiplication**: Integrates Apple's Accelerate framework's `cblas_sgemm` (BLAS) for highly optimized floating-point General Matrix Multiply (GEMM) operations
- **ARM NEON Vectorization**: Hand-optimized activation functions using ARM NEON intrinsics for significant speedup on ARM64 CPUs
- **Performance Benchmarking**: Includes `std::chrono`-based timing to accurately measure overall inference time and per-layer execution times
- **Command-Line Interface (CLI)**: Easy execution with configurable model paths, input files, and number of inference runs

## Architecture

The engine is built around a modular and extensible architecture:

### Core Components

- **Matrix.hpp/Matrix.cpp**: A fundamental class for 2D float arrays, serving as the primary data structure for inputs, weights, biases, and intermediate activations
  - Includes basic matrix operations and utility functions (e.g., `get_min_max`)
  - `multiply_accelerate`: Leverages `cblas_sgemm` for fast float matrix multiplication
  - `multiply_neon`: A custom NEON-optimized float matrix multiplication (primarily for benchmarking and learning)

- **Layer.hpp**: An abstract base class defining the forward interface, enabling polymorphism for different layer types

- **DenseLayer.hpp/DenseLayer.cpp**: Implements a fully connected layer
  - Stores float weights and biases
  - Includes `quantized_weights_data` for memory-efficient weight storage
  - The forward pass dequantizes int8 weights back to float and uses `multiply_accelerate` for computation
  - Handles weight quantization during model loading

- **ActivationLayer.hpp/ActivationLayer.cpp**: Implements element-wise activation functions (ReLU, Sigmoid)
  - The forward pass is heavily optimized using ARM NEON intrinsics for faster execution

- **NeuralNetwork.hpp/NeuralNetwork.cpp**: Manages the sequence of layers in the network
  - Provides `add_layer` and `predict` methods
  - `load_model_from_csv`: Handles loading weights and biases for each layer from CSV files
  - Includes a static helper `load_matrix_from_csv` for reading CSV data into Matrix objects

- **InferencePipeline.hpp/InferencePipeline.cpp**: A high-level class that orchestrates the entire inference process
  - Manages loading the full model and executing the forward pass

- **QuantizationUtils.hpp/QuantizationUtils.cpp**: Provides utility functions for float to int8 quantization and int8 to float dequantization

## Optimization Strategies & Performance

The core of this project's optimization lies in leveraging platform-specific hardware capabilities for common neural network operations.

### Accelerate Framework (BLAS)
- **Strategy**: For the computationally intensive matrix multiplication within DenseLayer, we utilize `cblas_sgemm` from Apple's Accelerate framework
- **Impact**: `cblas_sgemm` is often hand-tuned in assembly by Apple engineers, making it exceptionally fast by exploiting CPU features like SIMD and multi-threading

### ARM NEON Vectorization
- **Strategy**: For element-wise operations like activation functions, we implemented ARM NEON intrinsics
- **Impact**: This significantly reduced the execution time of `ActivationLayer::forward` to microseconds

### Quantization
- **Strategy**: Post-training quantization from float to int8 was applied to the network's weights
- **Impact**: 
  - Memory Efficiency: int8 weights reduce the model's memory footprint by 4x compared to float32
  - Performance: While int8 weights are dequantized back to float before computation, this still provides memory benefits

## Performance Metrics

After optimizing the dense layers with Accelerate and activation layers with NEON, the engine achieved the following performance on a typical Apple Silicon Mac (e.g., M1/M2/M3):

- **Network Size**: 3 layers (128 input features → 256 hidden neurons → 128 hidden neurons → 10 output classes)
- **Average Time per Inference**: 1.845 milliseconds

This demonstrates the effectiveness of combining a modular C++ design with platform-specific low-level optimizations for neural network inference.

## Challenges Faced & Solutions

### Initial `calculate_quant_params` Errors
- **Problem**: Early implementations resulted in drastically incorrect scale values, causing all quantized weights to saturate
- **Solution**: Deep dive into the mathematical formulas for asymmetric quantization, implementing robust linear mapping strategy

### Xcode Instruments Not Showing Application Hotspots
- **Problem**: Profiling tools showed only system-level calls, not application code
- **Solution**: Introduced random variation in input data to prevent compiler optimizations, relied on direct `std::chrono` timers

### `cblas_gemm_s8s8s32` Undeclared/BNNS Deprecation
- **Problem**: Attempt to use int8-specific BLAS function resulted in errors
- **Solution**: Pivoted to widely supported `cblas_sgemm` (float BLAS) for high performance with robust API

## Getting Started

### Prerequisites

- **macOS**: An Apple computer running macOS
- **Xcode**: Apple's integrated development environment (IDE)
- **Xcode Command Line Tools**: Run `xcode-select --install`
- **C++ Compiler**: Clang (comes with Xcode)
- **Python 3**: To generate dummy model CSVs

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/NNInferenceEngine.git
cd NNInferenceEngine
```

### 2. Generate Model CSVs

Create a Python file named `generate_model_csvs.py` in your project's root directory:

```python
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
```

Run the script:

```bash
python generate_model_csvs.py
```

### 3. Build the Project in Xcode

1. Open the `NNInferenceEngine.xcodeproj` file in Xcode
2. Go to **Product → Clean Build Folder** (hold Option key to see it)
3. Go to **Product → Build**

### 4. Copy Model to Build Folder

1. In Xcode's Project Navigator, navigate to the "Products" group
2. Right-click on **NNInferenceEngine** (the executable)
3. Select "Show in Finder"
4. Copy the `models_large` folder into this Debug directory

### 5. Run from Terminal

Navigate to the Debug directory and run:

```bash
# Navigate to your Debug directory (path will vary)
cd /Users/your-username/Library/Developer/Xcode/DerivedData/NNInferenceEngine-YOUR_UNIQUE_HASH/Build/Products/Debug/

# Run the program
# Arguments: <model_folder_path> <input_file_path> <num_inference_runs>
./NNInferenceEngine models_large models_large/input.csv 100
```

Expected output:
```
--- Model Loading ---
Loading Dense Layer 1 weights from: models_large/dense1_weights.csv
...
Model loaded with 6 layers.

--- Performance Benchmarking ---
Running inference 100 times for total time measurement...
Total inference time for 100 runs: 0.18453525 seconds
Average time per inference: 1.84535250 ms
```

## Using Your Own Model

### 1. Export Your Model

Train your model using PyTorch, TensorFlow, or any other framework, then export weights and biases to CSV format:

```python
import torch
import numpy as np
import os

# Example PyTorch model export
def export_model_to_csv(model, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    layer_idx = 1
    for name, param in model.named_parameters():
        if "weight" in name:
            # Transpose for C++ Matrix format (input_features, output_features)
            weights_np = param.data.numpy().T
            np.savetxt(os.path.join(output_folder, f"dense{layer_idx}_weights.csv"), 
                      weights_np, delimiter=',', fmt='%.8f')
        elif "bias" in name:
            biases_np = param.data.numpy().reshape(1, -1)
            np.savetxt(os.path.join(output_folder, f"dense{layer_idx}_biases.csv"), 
                      biases_np, delimiter=',', fmt='%.8f')
            layer_idx += 1
```

### 2. Update Model Architecture

Modify `NeuralNetwork::load_model_from_csv` in `NeuralNetwork.cpp` to match your model's:
- Number of layers
- Layer dimensions
- Activation functions

### 3. Prepare Input Data

Create a CSV file with your input data (1 row, N columns where N = input features).

### 4. Run with Custom Model

```bash
./NNInferenceEngine your_model_folder your_model_folder/input.csv 100
```

## Future Work

- **Full int8 Inference Pipeline**: Implement true int8 GEMM using ARM NEON intrinsics
- **More Layer Types**: Convolutional layers, pooling layers, batch normalization
- **Model Serialization**: Custom binary format for faster loading
- **Training Support**: Implement backpropagation and optimizers
- **Batch Inference**: Process multiple inputs in parallel
- **Cross-Platform Support**: Abstract platform-specific optimizations
- **GPU Acceleration**: Integrate Metal Performance Shaders (MPS Graph)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Apple's Accelerate framework for high-performance BLAS operations
- ARM NEON intrinsics for vectorized computations
- The broader machine learning community for inspiration and best practices
