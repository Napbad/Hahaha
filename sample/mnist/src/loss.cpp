#include "loss.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <iostream>

namespace mnist {

std::unique_ptr<Loss> createLoss(LossType type) {
    switch (type) {
        case LossType::MSE:
            return std::make_unique<MSELoss>();
        case LossType::CrossEntropy:
            return std::make_unique<CrossEntropyLoss>();
        case LossType::CategoricalCrossEntropy:
            return std::make_unique<CategoricalCrossEntropyLoss>();
        default:
            printf("int ");
            throw std::invalid_argument("Unknown loss type");
    }
}

// MSELoss implementation
float MSELoss::compute(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Compute squared differences
    Eigen::MatrixXf squared_diff = (predictions - targets).array().square();
    
    // Average over all elements
    return squared_diff.sum() / predictions.size();
}

Eigen::MatrixXf MSELoss::gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Gradient of MSE is 2 * (predictions - targets) / n
    return 2.0f * (predictions - targets) / predictions.size();
}

// CrossEntropyLoss implementation
float CrossEntropyLoss::compute(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Small epsilon to avoid log(0)
    const float epsilon = 1e-7f;
    
    // Element-wise multiplication and sum
    Eigen::MatrixXf safe_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
    Eigen::MatrixXf log_probs = safe_preds.array().log();
    float loss = -(targets.array() * log_probs.array()).sum() / predictions.rows();
    
    return loss;
}

Eigen::MatrixXf CrossEntropyLoss::gradient(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
    // Small epsilon to avoid division by zero
    const float epsilon = 1e-7f;
    
    // Gradient is -targets / predictions
    Eigen::MatrixXf safe_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
    return -targets.array() / safe_preds.array() / predictions.rows();
}

// CategoricalCrossEntropyLoss implementation
Eigen::MatrixXf CategoricalCrossEntropyLoss::softmax(const Eigen::MatrixXf& x) {
    // Apply softmax row-wise
    Eigen::MatrixXf result(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); ++i) {
        // Subtract max for numerical stability
        Eigen::ArrayXf row = x.row(i).array();
        float max_val = row.maxCoeff();
        row = (row - max_val).exp();
        result.row(i) = row / row.sum();
    }
    
    return result;
}

float CategoricalCrossEntropyLoss::compute(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) {
    // Apply softmax to get probabilities
    m_last_softmax = softmax(logits);
    
    // Small epsilon to avoid log(0)
    const float epsilon = 1e-7f;
    
    // Compute cross-entropy loss
    Eigen::MatrixXf safe_probs = m_last_softmax.array().max(epsilon).min(1.0f - epsilon);
    float loss = -(targets.array() * safe_probs.array().log()).sum() / logits.rows();
    
    return loss;
}

Eigen::MatrixXf CategoricalCrossEntropyLoss::gradient(const Eigen::MatrixXf& logits, const Eigen::MatrixXf& targets) {
    // Gradient of softmax CE is (softmax(logits) - targets) / batch_size
    Eigen::MatrixXf probs = softmax(logits);
    return (probs - targets) / logits.rows();
}

} // namespace mnist
