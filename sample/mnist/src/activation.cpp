#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mnist {

std::unique_ptr<Activation> createActivation(ActivationType type) {
    switch (type) {
        case ActivationType::Identity:
            return std::make_unique<IdentityActivation>();
        case ActivationType::Sigmoid:
            return std::make_unique<SigmoidActivation>();
        case ActivationType::ReLU:
            return std::make_unique<ReLUActivation>();
        case ActivationType::Tanh:
            return std::make_unique<TanhActivation>();
        case ActivationType::Softmax:
            return std::make_unique<SoftmaxActivation>();
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

// IdentityActivation implementation
Eigen::MatrixXf IdentityActivation::forward(const Eigen::MatrixXf& x) {
    return x;
}

Eigen::MatrixXf IdentityActivation::backward(const Eigen::MatrixXf& grad_output, const Eigen::MatrixXf&) {
    return grad_output;
}

// SigmoidActivation implementation
Eigen::MatrixXf SigmoidActivation::forward(const Eigen::MatrixXf& x) {
    return 1.0f / (1.0f + (-x.array()).exp());
}

Eigen::MatrixXf SigmoidActivation::backward(const Eigen::MatrixXf& grad_output, const Eigen::MatrixXf& output) {
    // Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    return grad_output.array() * output.array() * (1.0f - output.array());
}

// ReLUActivation implementation
Eigen::MatrixXf ReLUActivation::forward(const Eigen::MatrixXf& x) {
    // Save mask for backward pass
    m_mask = (x.array() > 0.0f).cast<float>();
    return x.cwiseMax(0.0f);
}

Eigen::MatrixXf ReLUActivation::backward(const Eigen::MatrixXf& grad_output, const Eigen::MatrixXf&) {
    // Element-wise multiplication with the mask
    return grad_output.array() * m_mask.array();
}

// TanhActivation implementation
Eigen::MatrixXf TanhActivation::forward(const Eigen::MatrixXf& x) {
    return x.array().tanh();
}

Eigen::MatrixXf TanhActivation::backward(const Eigen::MatrixXf& grad_output, const Eigen::MatrixXf& output) {
    // Derivative of tanh is 1 - tanh²(x)
    return grad_output.array() * (1.0f - output.array().square());
}

// SoftmaxActivation implementation
Eigen::MatrixXf SoftmaxActivation::forward(const Eigen::MatrixXf& x) {
    // Compute softmax for each row (sample) separately
    const int rows = x.rows();
    const int cols = x.cols();
    Eigen::MatrixXf result(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        // Subtract max for numerical stability
        Eigen::ArrayXf row = x.row(i).array();
        float max_val = row.maxCoeff();
        row = (row - max_val).exp();
        result.row(i) = row / row.sum();
    }
    
    // Store result for backward pass
    m_output = result;
    return result;
}

Eigen::MatrixXf SoftmaxActivation::backward(const Eigen::MatrixXf& grad_output, const Eigen::MatrixXf&) {
    // Note: This is a simplified Jacobian-vector product assuming cross-entropy loss
    // For other loss functions, the full Jacobian would be needed
    return grad_output;
}

} // namespace mnist
