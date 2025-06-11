//
//  main.cpp
//  NNInferenceEngine
//
//  Created by Shrey Sharma on 10/06/25.
//

#include <iostream>
#include "Matrix.hpp"


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
    return 0;
}
