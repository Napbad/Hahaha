#pragma once

#include <Eigen/Dense>
#include <memory>

namespace mnist
{

/**
 * @brief Enumeration of supported activation function types
 */
enum class ActivationType
{
    Identity, ///< f(x) = x
    Sigmoid,  ///< f(x) = 1 / (1 + e^(-x))
    ReLU,     ///< f(x) = max(0, x)
    Tanh,     ///< f(x) = tanh(x)
    Softmax   ///< f(x_i) = e^(x_i) / sum(e^(x_j))
};

/**
 * @brief Abstract base class for activation functions
 *
 * This class defines the interface for activation functions.
 */
class Activation
{
  public:
    virtual ~Activation() = default;

    /**
     * @brief Apply the activation function
     *
     * @param x Input tensor
     * @return Output tensor after activation
     */
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& x) = 0;

    /**
     * @brief Compute gradient of the activation function
     *
     * @param grad_output Gradient from the next layer
     * @param output Output of the forward pass
     * @return Gradient to pass to the previous layer
     */
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                                     const Eigen::MatrixXf& output) = 0;
};

/**
 * @brief Identity activation function (no transformation)
 */
class IdentityActivation : public Activation
{
  public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                             const Eigen::MatrixXf& output) override;
};

/**
 * @brief Sigmoid activation function
 *
 * f(x) = 1 / (1 + e^(-x))
 */
class SigmoidActivation : public Activation
{
  public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                             const Eigen::MatrixXf& output) override;
};

/**
 * @brief ReLU activation function
 *
 * f(x) = max(0, x)
 */
class ReLUActivation : public Activation
{
  public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                             const Eigen::MatrixXf& output) override;

  private:
    Eigen::MatrixXf m_mask; // Stores which elements were > 0 in forward pass
};

/**
 * @brief Tanh activation function
 *
 * f(x) = tanh(x)
 */
class TanhActivation : public Activation
{
  public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                             const Eigen::MatrixXf& output) override;
};

/**
 * @brief Softmax activation function
 *
 * f(x_i) = e^(x_i) / sum(e^(x_j))
 */
class SoftmaxActivation : public Activation
{
  public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad_output,
                             const Eigen::MatrixXf& output) override;

  private:
    Eigen::MatrixXf m_output; // Cached output for backward pass
};

/**
 * @brief Create an activation function based on the given type
 *
 * @param type Type of activation function to create
 * @return Unique pointer to the activation function
 */
std::unique_ptr<Activation> createActivation(ActivationType type);

} // namespace mnist
