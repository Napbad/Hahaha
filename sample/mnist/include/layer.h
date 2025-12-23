#pragma once

#include <memory>
#include <Eigen/Dense>
#include "activation.h"

namespace mnist {

/**
 * @brief Base class for neural network layers
 * 
 * This abstract class defines the interface for all neural network layers.
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * @brief Forward pass through the layer
     * 
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
    
    /**
     * @brief Backward pass through the layer
     * 
     * @param grad_output Gradient from the next layer
     * @return Gradient to pass to the previous layer
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) = 0;
    
    /**
     * @brief Update layer parameters using calculated gradients
     * 
     * @param learning_rate Learning rate for parameter updates
     */
    virtual void update(float learning_rate) = 0;

    /**
     * @brief Get the number of parameters in this layer
     * 
     * @return Number of trainable parameters
     */
    virtual size_t getParameterCount() const = 0;
};

/**
 * @brief Fully connected (dense) layer implementation
 * 
 * This layer implements a fully connected layer with optional activation.
 */
class DenseLayer : public Layer {
public:
    /**
     * @brief Construct a dense layer
     * 
     * @param input_size Number of input features
     * @param output_size Number of output features
     * @param activation_type Type of activation function to use
     */
    DenseLayer(size_t input_size, size_t output_size, ActivationType activation_type = ActivationType::ReLU);
    
    /**
     * @brief Forward pass through the layer
     * 
     * @param input Input tensor of shape (batch_size, input_size)
     * @return Output tensor of shape (batch_size, output_size)
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    
    /**
     * @brief Backward pass through the layer
     * 
     * @param grad_output Gradient from the next layer of shape (batch_size, output_size)
     * @return Gradient to pass to the previous layer of shape (batch_size, input_size)
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output) override;
    
    /**
     * @brief Update layer weights and biases
     * 
     * @param learning_rate Learning rate for parameter updates
     */
    void update(float learning_rate) override;
    
    /**
     * @brief Get the number of parameters in this layer
     * 
     * @return Number of trainable parameters (weights + biases)
     */
    size_t getParameterCount() const override;
    
private:
    // Layer dimensions
    size_t m_input_size;
    size_t m_output_size;
    
    // Layer parameters and their gradients
    Eigen::MatrixXf m_weights;  // Shape: (input_size, output_size)
    Eigen::VectorXf m_biases;   // Shape: (output_size)
    Eigen::MatrixXf m_dweights; // Gradients for weights
    Eigen::VectorXf m_dbiases;  // Gradients for biases
    
    // Cached values for backpropagation
    Eigen::MatrixXf m_last_input;      // Last input from forward pass
    Eigen::MatrixXf m_last_output;     // Last output before activation
    Eigen::MatrixXf m_activated_output; // Last output after activation
    
    // Activation function
    std::unique_ptr<Activation> m_activation;
    
    /**
     * @brief Initialize layer weights using Xavier/Glorot initialization
     */
    void initializeWeights();
};

} // namespace mnist
