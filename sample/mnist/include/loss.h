#pragma once

#include <Eigen/Dense>
#include <memory>

namespace mnist {

/**
 * @brief Enumeration of supported loss function types
 */
enum class LossType {
    MSE,               ///< Mean Squared Error
    CrossEntropy,      ///< Cross Entropy Loss
    CategoricalCrossEntropy  ///< Categorical Cross Entropy Loss (with softmax)
};

/**
 * @brief Abstract base class for loss functions
 * 
 * This class defines the interface for loss functions.
 */
class Loss {
public:
    virtual ~Loss() = default;
    
    /**
     * @brief Compute the loss between predictions and targets
     * 
     * @param predictions Model predictions
     * @param targets Ground truth values
     * @return Scalar loss value
     */
    virtual float compute(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) = 0;
    
    /**
     * @brief Compute gradient of the loss with respect to predictions
     * 
     * @param predictions Model predictions
     * @param targets Ground truth values
     * @return Gradient tensor of the same shape as predictions
     */
    virtual Eigen::MatrixXf gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) = 0;
};

/**
 * @brief Mean Squared Error loss function
 * 
 * L = (1/n) * sum((predictions - targets)^2)
 */
class MSELoss : public Loss {
public:
    float compute(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
};

/**
 * @brief Cross Entropy loss function
 * 
 * L = -sum(targets * log(predictions))
 * 
 * Note: Assumes predictions are already in probability space (e.g., after sigmoid or softmax)
 */
class CrossEntropyLoss : public Loss {
public:
    float compute(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) override;
};

/**
 * @brief Categorical Cross Entropy loss function with integrated softmax
 * 
 * This loss function combines softmax activation with cross entropy loss for numerical stability.
 */
class CategoricalCrossEntropyLoss : public Loss {
public:
    float compute(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) override;
    Eigen::MatrixXf gradient(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) override;
private:
    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x);
    Eigen::MatrixXf m_last_softmax; // Cache for gradient computation
};

/**
 * @brief Create a loss function based on the given type
 * 
 * @param type Type of loss function to create
 * @return Unique pointer to the loss function
 */
std::unique_ptr<Loss> createLoss(LossType type);

} // namespace mnist
