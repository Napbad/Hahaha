#include "network.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

namespace mnist {

Network& Network::addLayer(size_t input_size, size_t output_size, ActivationType activation_type) {
    m_layers.push_back(std::make_unique<DenseLayer>(input_size, output_size, activation_type));
    return *this;
}

Network& Network::compile(LossType loss_type, OptimizerType optimizer_type, float learning_rate) {
    // Create loss function
    m_loss = createLoss(loss_type);
    
    // Create optimizer
    m_optimizer = createOptimizer(optimizer_type, learning_rate);
    
    m_is_compiled = true;
    return *this;
}

Eigen::MatrixXf Network::forward(const Eigen::MatrixXf& input) {
    if (m_layers.empty()) {
        return input;
    }
    
    // Forward through each layer
    Eigen::MatrixXf output = input;
    for (auto& layer : m_layers) {
        output = layer->forward(output);
    }
    
    return output;
}

float Network::trainOnBatch(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets) {
    if (!m_is_compiled) {
        throw std::runtime_error("Network not compiled. Call compile() before training.");
    }
    
    // Forward pass
    Eigen::MatrixXf predictions = forward(inputs);
    
    // Compute loss
    float loss = m_loss->compute(predictions, targets);
    
    // Compute initial gradient from loss function
    Eigen::MatrixXf gradient = m_loss->gradient(predictions, targets);
    
    // Backward pass through each layer
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        gradient = (*it)->backward(gradient);
    }
    
    // Update weights using optimizer
    m_optimizer->update(m_layers);
    
    return loss;
}

std::pair<float, float> Network::evaluate(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets) {
    // Forward pass
    Eigen::MatrixXf predictions = forward(inputs);
    
    // Compute loss
    float loss = m_loss->compute(predictions, targets);
    
    // Compute accuracy
    int correct = 0;
    for (int i = 0; i < predictions.rows(); ++i) {
        // Find index of maximum value in predictions and targets
        Eigen::Index pred_idx, target_idx;
        predictions.row(i).maxCoeff(&pred_idx);
        targets.row(i).maxCoeff(&target_idx);
        
        if (pred_idx == target_idx) {
            ++correct;
        }
    }
    
    float accuracy = static_cast<float>(correct) / inputs.rows();
    
    return {loss, accuracy};
}

Eigen::VectorXi Network::predict(const Eigen::MatrixXf& inputs) {
    // Forward pass
    Eigen::MatrixXf predictions = forward(inputs);
    
    // Find class with highest probability for each sample
    Eigen::VectorXi result(inputs.rows());
    for (int i = 0; i < predictions.rows(); ++i) {
        Eigen::Index idx;
        predictions.row(i).maxCoeff(&idx);
        result(i) = static_cast<int>(idx);
    }
    
    return result;
}

bool Network::save(const std::string& filepath) const {
    // Implement model saving functionality
    // This would typically serialize layer weights and architecture
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for saving: " << filepath << std::endl;
        return false;
    }
    
    // For a real implementation, we would serialize the network parameters
    // For now, this is a placeholder
    
    return true;
}

bool Network::load(const std::string& filepath) {
    // Implement model loading functionality
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for loading: " << filepath << std::endl;
        return false;
    }
    
    // For a real implementation, we would deserialize the network parameters
    // For now, this is a placeholder
    
    return true;
}

size_t Network::getParameterCount() const {
    size_t count = 0;
    for (const auto& layer : m_layers) {
        count += layer->getParameterCount();
    }
    return count;
}

void Network::summary() const {
    std::cout << "==================================================================" << std::endl;
    std::cout << "Neural Network Summary" << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << std::left << std::setw(30) << "Layer" 
              << std::setw(15) << "Output Shape" 
              << std::setw(15) << "Parameters" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    size_t total_params = 0;
    for (size_t i = 0; i < m_layers.size(); ++i) {
        std::string layer_name = "Dense_" + std::to_string(i + 1);
        std::string output_shape = "?"; // In a real implementation, we would get this from the layer
        size_t params = m_layers[i]->getParameterCount();
        total_params += params;
        
        std::cout << std::left << std::setw(30) << layer_name 
                  << std::setw(15) << output_shape 
                  << std::setw(15) << params << std::endl;
    }
    
    std::cout << "==================================================================" << std::endl;
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "==================================================================" << std::endl;
}

} // namespace mnist
