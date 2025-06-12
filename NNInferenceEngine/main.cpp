//
//  main.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#include <iostream>
#include "Matrix.hpp"
#include <chrono>


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

    
    return 0;
}
