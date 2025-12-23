#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace mnist {

class Layer;

/**
 * @brief Enumeration of supported optimizer types
 */
enum class OptimizerType {
    SGD,       ///< Stochastic Gradient Descent
    SGDMomentum, ///< SGD with Momentum
    Adam       ///< Adam Optimizer
};

/**
 * @brief Abstract base class for optimizers
 * 
 * This class defines the interface for all optimizers.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * @brief Update layer parameters using calculated gradients
     * 
     * @param layers Vector of layers to update
     */
    virtual void update(std::vector<std::unique_ptr<Layer>>& layers) = 0;
};

/**
 * @brief Simple Stochastic Gradient Descent optimizer
 */
class SGDOptimizer : public Optimizer {
public:
    /**
     * @brief Construct an SGD optimizer
     * 
     * @param learning_rate Learning rate for gradient descent
     */
    explicit SGDOptimizer(float learning_rate);
    
    /**
     * @brief Update layer parameters using calculated gradients
     * 
     * @param layers Vector of layers to update
     */
    void update(std::vector<std::unique_ptr<Layer>>& layers) override;
    
    /**
     * @brief Get the learning rate
     * 
     * @return Current learning rate
     */
    float getLearningRate() const { return m_learning_rate; }
    
    /**
     * @brief Set the learning rate
     * 
     * @param learning_rate New learning rate
     */
    void setLearningRate(float learning_rate) { m_learning_rate = learning_rate; }
    
private:
    float m_learning_rate;
};

/**
 * @brief SGD with Momentum optimizer
 */
class SGDMomentumOptimizer : public Optimizer {
public:
    /**
     * @brief Construct an SGD with Momentum optimizer
     * 
     * @param learning_rate Learning rate for gradient descent
     * @param momentum Momentum coefficient (typically 0.9)
     */
    SGDMomentumOptimizer(float learning_rate, float momentum = 0.9);
    
    /**
     * @brief Update layer parameters using calculated gradients
     * 
     * @param layers Vector of layers to update
     */
    void update(std::vector<std::unique_ptr<Layer>>& layers) override;
    
private:
    float m_learning_rate;
    float m_momentum;
    std::vector<std::vector<Eigen::MatrixXf>> m_velocity; // Velocity for each parameter
};

/**
 * @brief Adam optimizer
 * 
 * Implementation of the Adam algorithm (Adaptive Moment Estimation)
 * as described in https://arxiv.org/abs/1412.6980
 */
class AdamOptimizer : public Optimizer {
public:
    /**
     * @brief Construct an Adam optimizer
     * 
     * @param learning_rate Learning rate
     * @param beta1 Exponential decay rate for first moment estimates (default: 0.9)
     * @param beta2 Exponential decay rate for second moment estimates (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     */
    AdamOptimizer(float learning_rate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    
    /**
     * @brief Update layer parameters using calculated gradients
     * 
     * @param layers Vector of layers to update
     */
    void update(std::vector<std::unique_ptr<Layer>>& layers) override;
    
private:
    float m_learning_rate;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    unsigned int m_t = 0; // Time step counter
    
    // First moment estimates (mean)
    std::vector<std::vector<Eigen::MatrixXf>> m_m;
    
    // Second moment estimates (variance)
    std::vector<std::vector<Eigen::MatrixXf>> m_v;
};

/**
 * @brief Create an optimizer based on the given type
 * 
 * @param type Type of optimizer to create
 * @param learning_rate Learning rate for the optimizer
 * @return Unique pointer to the optimizer
 */
std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, float learning_rate);

} // namespace mnist
