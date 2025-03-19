#ifndef HEADERS_ML
#define HEADERS_ML

#include <iostream>
#include <vector>
#include <cassert>
#include <random>

#include <algorithm> // For std::max

// ReLU activation function
std::vector<double> relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::max(0.0, x[i]);
    }
    return result;
}

// Derivative of ReLU (for backpropagation)
std::vector<double> relu_derivative(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = (x[i] > 0) ? 1.0 : 0.0;
    }
    return result;
}

#endif // !HEADERS_ML