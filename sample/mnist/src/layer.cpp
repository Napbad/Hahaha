#include "layer.h"
#include <random>
#include <cmath>

namespace mnist {

DenseLayer::DenseLayer(size_t input_size, size_t output_size, ActivationType activation_type)
    : m_input_size(input_size), 
      m_output_size(output_size),
      m_weights(input_size, output_size),
      m_biases(output_size),
      m_dweights(input_size, output_size),
      m_dbiases(output_size),
      m_activation(createActivation(activation_type)) {
    
    // Initialize weights and biases
    initializeWeights();
    m_biases.setZero();
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    // Cache input for backward pass
    m_last_input = input;
    
    // Linear transformation: y = x * W + b
    m_last_output = input * m_weights;
    
    // Add bias term
    for (int i = 0; i < m_last_output.rows(); ++i) {
        m_last_output.row(i) += m_biases.transpose();
    }
    
    // Apply activation function
    m_activated_output = m_activation->forward(m_last_output);
    
    return m_activated_output;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& grad_output) {
    // Backpropagate through activation function
    Eigen::MatrixXf d_linear = m_activation->backward(grad_output, m_activated_output);
    
    // Compute gradients
    m_dweights = m_last_input.transpose() * d_linear;
    m_dbiases = d_linear.colwise().sum();
    
    // Backpropagate to previous layer
    return d_linear * m_weights.transpose();
}

void DenseLayer::update(float learning_rate) {
    // Update weights and biases using gradients
    m_weights -= learning_rate * m_dweights;
    m_biases -= learning_rate * m_dbiases;
}

size_t DenseLayer::getParameterCount() const {
    // Number of weights + number of biases
    return m_input_size * m_output_size + m_output_size;
}

void DenseLayer::initializeWeights() {
    // Xavier/Glorot initialization
    float limit = std::sqrt(6.0f / (m_input_size + m_output_size));
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    // Initialize weights
    for (int i = 0; i < m_weights.rows(); ++i) {
        for (int j = 0; j < m_weights.cols(); ++j) {
            m_weights(i, j) = dis(gen);
        }
    }
}

} // namespace mnist
