//
//  main.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#include <iostream>
#include "Matrix.hpp"
#include <chrono>
#include <arm_neon.h>  //including neon intrinsic architechture
#include <numeric>
#include "VectorOps.h"


int main(){
    std::cout << "Matrix Multiplication";
    
    //testing basic matrix creation and access
    Matrix m1(2, 3);
    m1.set_value(0, 0, 1.0f);
    m1.set_value(0, 1, 2.0f);
    m1.set_value(0, 2, 3.0f);
    m1.set_value(1, 0, 4.0f);
    m1.set_value(1, 1, 5.0f);
    m1.set_value(1, 2, 6.0f);
    
    std::cout << "\nMatrix m1:\n";
    m1.print();
    
    std::cout << "Value at (0,1): " << m1.get_value(0, 1) << std::endl;
    
    // Test error handling for out-of-bounds access
    try {
        m1.get_value(2, 0); // This should throw an exception
        } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        }
    
    //testing addition
    Matrix m2(2, 3);
    m2.set_value(0, 0, 1.0f); m2.set_value(0, 1, 1.0f); m2.set_value(0, 2, 1.0f);
    m2.set_value(1, 0, 1.0f); m2.set_value(1, 1, 1.0f); m2.set_value(1, 2, 1.0f);
        std::cout << "\nMatrix m2:\n";
        m2.print();

        try {
            Matrix m_sum = add(m1, m2);
            std::cout << "\nm1 + m2:\n";
            m_sum.print(); // Expected: 2 3 4 / 5 6 7
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    
    // Test Multiplication
        Matrix A(2, 2);
        A.set_value(0, 0, 1.0f); A.set_value(0, 1, 2.0f);
        A.set_value(1, 0, 3.0f); A.set_value(1, 1, 4.0f);
        std::cout << "\nMatrix A:\n";
        A.print();

        Matrix B(2, 2);
        B.set_value(0, 0, 5.0f); B.set_value(0, 1, 6.0f);
        B.set_value(1, 0, 7.0f); B.set_value(1, 1, 8.0f);
        std::cout << "\nMatrix B:\n";
        B.print();

        try {
            Matrix C = multiply(A, B);
            std::cout << "\nA * B:\n";
            C.print(); // Expected: 19 22 / 43 50
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    
    int size = 500;
    Matrix LargeA(size, size);
    Matrix LargeB(size, size);
    
    //populating with some random values
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            LargeA.set_value(i, j, static_cast<float>(i + j));
            LargeB.set_value(i , j, static_cast<float>(i * j));
        }
    }
    
    std::cout << "\nBenchMarking " << size << "x" << size << "Matrix Multiplication..\n";
    
    //manual multiplication
    auto start_manual = std::chrono::high_resolution_clock::now();
    Matrix result_manual = multiply(LargeA, LargeB);
    auto end_manual = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_manual = end_manual - start_manual;
    std::cout<<"Manual Matrix Multiplication time: " << duration_manual.count() << "seconds\n";
    
    //Accelerate Multiplication
    auto start_accelerate = std::chrono::high_resolution_clock::now();
    Matrix result_accelerate = multiply_accelerate(LargeA, LargeB);
    auto end_accelerate = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_accelerate = end_accelerate - start_accelerate;
    std::cout << "Accelerate multiplication time: " << duration_accelerate.count() << " seconds\n";
    
    // Optional: Verify results are approximately equal
    float max_diff = 0.0f;
    for (int i = 0; i < size * size; ++i) {
    float diff = std::abs(result_manual.data[i] - result_accelerate.data[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    std::cout << "Max difference between manual and Accelerate results: " << max_diff << std::endl;
    
    //SIMD Fundamentals - NEON Experiment
    float a_arr[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_arr[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result_arr[4];
    
    //1. Loading data from memory into NEON registers(foat32x4_t)
    // vld1q_32 loads 4 floats into a 128 bit register
    float32x4_t vec_a = vld1q_f32(a_arr);
    float32x4_t vec_b = vld1q_f32(b_arr);
    
    //2. performing elementwise addition using NEON intrinsic
    // vaddq_f32 add corresponding elements of two 128-bit float vecotrs
    float32x4_t vec_result = vaddq_f32(vec_a, vec_b);
    
    //3. storing the result back to memory from NEON registers
    // vst1q_f32 stores the 4 floats from the 128-bit registers back into memory
    vst1q_f32(result_arr, vec_result);
    
    std::cout << "Result of NEON vector addition: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << result_arr[i] << " ";
        }
    std::cout << std::endl;
    
    std::cout << "First element of vec_result: " << vgetq_lane_f32(vec_result, 0) << std::endl;
    
    //SIMD Neon vector operation with benchmarking with scalar operation
    int vector_size = 1000000;
    
    std::vector<float> a(vector_size);
    std::vector<float> b(vector_size);
    std::vector<float> result_scalar(vector_size);
    std::vector<float> result_neon(vector_size);
    
    //initializing vectors
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 10.0f);
    
    std::cout << "\nBenchmarking vector addition for " << vector_size << " elements...\n";
    
    //scalar vector addition
    auto start_scalar = std::chrono::high_resolution_clock::now();
    vector_add_scalar(a.data(), b.data(), result_scalar.data(), vector_size);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_scalar = end_scalar - start_scalar;
    std::cout << "Scalar vector addition time: " << duration_scalar.count() << "sec\n";
    
    //NEON vector additon
    auto start_neon = std::chrono::high_resolution_clock::now();
    vector_add_neon(a.data(), b.data(), result_neon.data(), vector_size);
    auto end_neon = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_neon = end_neon - start_neon;
    std::cout << "NEON vector addition time: " << duration_neon.count() << " sec\n";
    
    //Verifying results for correctness
    float max_diff_neon = 0.0f;
    for (int i = 0; i < vector_size; ++i){
        float diff = std::abs(result_scalar[i] - result_neon[i]);
        if (diff>max_diff_neon){
            max_diff_neon = diff;
        }
    }
    
    std::cout << "\nBenchmarking " << size << "x" << size << " Matrix Multiplication...\n";
    
    auto start_neon2 = std::chrono::high_resolution_clock::now();
    Matrix result_neon2 = multiply_neon(LargeA, LargeB);
    auto end_neon2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_neon2 = end_neon2 - start_neon2;
    std::cout << "NEON multiplication time: " << duration_neon2.count() << " seconds\n";
    
    //verifying neon result against manual and accelerate
    float max_diff_neon2 = 0.0f;
    for (int i = 0; i < size*size; ++i){
        float diff = std::abs(result_manual.data[i] - result_neon2.data[i]);
        if (diff>max_diff){
            max_diff_neon2 = diff;
        }
    }
    std::cout << "Max difference between manual and NEON results: " << max_diff_neon2 << std::endl;
    
    return 0;
}
