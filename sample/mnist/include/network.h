#pragma once

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "layer.h"
#include "loss.h"
#include "optimizer.h"

namespace mnist {

/**
 * @brief Neural Network class for MNIST digit recognition
 * 
 * This class implements a feedforward neural network with configurable layers.
 */
class Network {
public:
    /**
     * @brief Default constructor
     */
    Network() = default;
    
    /**
     * @brief Add a dense layer to the network
     * 
     * @param input_size Number of input features
     * @param output_size Number of output features
     * @param activation_type Type of activation function to use
     * @return Reference to this network (for method chaining)
     */
    Network& addLayer(size_t input_size, size_t output_size, 
                      ActivationType activation_type = ActivationType::ReLU);
    
    /**
     * @brief Compile the network with loss function and optimizer
     * 
     * @param loss_type Type of loss function to use
     * @param optimizer_type Type of optimizer to use
     * @param learning_rate Initial learning rate
     * @return Reference to this network (for method chaining)
     */
    Network& compile(LossType loss_type = LossType::CategoricalCrossEntropy, 
                     OptimizerType optimizer_type = OptimizerType::Adam, 
                     float learning_rate = 0.001);
    
    /**
     * @brief Forward pass through the network
     * 
     * @param input Input tensor of shape (batch_size, input_features)
     * @return Output tensor of shape (batch_size, output_features)
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
    /**
     * @brief Perform a training step on a batch of data
     * 
     * @param inputs Batch input tensor of shape (batch_size, input_features)
     * @param targets Batch target tensor of shape (batch_size, output_features)
     * @return Loss value for this batch
     */
    float trainOnBatch(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets);
    
    /**
     * @brief Evaluate the network on test data
     * 
     * @param inputs Test input tensor of shape (n_samples, input_features)
     * @param targets Test target tensor of shape (n_samples, output_features)
     * @return std::pair<float, float> containing (loss, accuracy)
     */
    std::pair<float, float> evaluate(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets);
    
    /**
     * @brief Predict class labels for input data
     * 
     * @param inputs Input tensor of shape (n_samples, input_features)
     * @return Eigen::VectorXi containing predicted class indices
     */
    Eigen::VectorXi predict(const Eigen::MatrixXf& inputs);
    
    /**
     * @brief Save the network parameters to a file
     * 
     * @param filepath Path to save the network parameters
     * @return true if successful
     */
    bool save(const std::string& filepath) const;
    
    /**
     * @brief Load network parameters from a file
     * 
     * @param filepath Path to load the network parameters from
     * @return true if successful
     */
    bool load(const std::string& filepath);
    
    /**
     * @brief Get the total number of parameters in the network
     * 
     * @return Total number of parameters
     */
    size_t getParameterCount() const;
    
    /**
     * @brief Print a summary of the network architecture
     */
    void summary() const;
    
private:
    std::vector<std::unique_ptr<Layer>> m_layers;
    std::unique_ptr<Loss> m_loss;
    std::unique_ptr<Optimizer> m_optimizer;
    bool m_is_compiled = false;
};

} // namespace mnist
