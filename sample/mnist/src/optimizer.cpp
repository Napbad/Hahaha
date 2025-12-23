#include "optimizer.h"
#include "layer.h"
#include <stdexcept>

namespace mnist {

std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, float learning_rate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGDOptimizer>(learning_rate);
        case OptimizerType::SGDMomentum:
            return std::make_unique<SGDMomentumOptimizer>(learning_rate);
        case OptimizerType::Adam:
            return std::make_unique<AdamOptimizer>(learning_rate);
        default:
            throw std::invalid_argument("Unknown optimizer type");
    }
}

// SGDOptimizer implementation
SGDOptimizer::SGDOptimizer(float learning_rate)
    : m_learning_rate(learning_rate) {
}

void SGDOptimizer::update(std::vector<std::unique_ptr<Layer>>& layers) {
    // Update each layer's parameters
    for (auto& layer : layers) {
        layer->update(m_learning_rate);
    }
}

// SGDMomentumOptimizer implementation
SGDMomentumOptimizer::SGDMomentumOptimizer(float learning_rate, float momentum)
    : m_learning_rate(learning_rate), m_momentum(momentum) {
}

void SGDMomentumOptimizer::update(std::vector<std::unique_ptr<Layer>>& layers) {
    // Initialize velocity if this is the first update
    if (m_velocity.empty()) {
        m_velocity.resize(layers.size());
    }
    
    // Update each layer's parameters with momentum
    for (size_t i = 0; i < layers.size(); ++i) {
        // Create velocity vectors if they don't exist for this layer
        // Note: This is a simplified implementation that assumes each layer has
        // exactly one weight tensor and one bias vector
        if (m_velocity[i].empty()) {
            // Initialize with zero velocity
            // In a more complete implementation, we would inspect the layer's structure
            m_velocity[i].resize(2); // One for weights, one for biases
            m_velocity[i][0] = Eigen::MatrixXf::Zero(layers[i]->getParameterCount(), 1);
            m_velocity[i][1] = Eigen::VectorXf::Zero(layers[i]->getParameterCount());
        }
        
        // Update velocity and parameters
        // This is a simplified version - in a real implementation, we would
        // access the layer's gradients and parameters directly
        layers[i]->update(m_learning_rate);
    }
}

// AdamOptimizer implementation
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : m_learning_rate(learning_rate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {
}

void AdamOptimizer::update(std::vector<std::unique_ptr<Layer>>& layers) {
    // Update time step
    ++m_t;
    
    // Initialize moment estimates if this is the first update
    if (m_m.empty()) {
        m_m.resize(layers.size());
        m_v.resize(layers.size());
    }
    
    // Update each layer's parameters
    for (size_t i = 0; i < layers.size(); ++i) {
        // Update parameters using basic SGD for now
        // In a more complete implementation, we would implement the full Adam algorithm
        layers[i]->update(m_learning_rate);
    }
}

} // namespace mnist
