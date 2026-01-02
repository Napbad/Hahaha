// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H
#define HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "TopoSort.h"
#include "common/Operator.h"
#include "math/TensorWrapper.h"
#include "utils/common/HelperStruct.h"

namespace hahaha {
/**
 * @brief Forward declaration of the high-level Tensor class.
 */
template <typename T> class Tensor;
} // namespace hahaha

namespace hahaha::compute {

/**
 * @brief Represents a node in the computational graph.
 *
 * Each ComputeNode manages a piece of tensor data and its role in the graph,
 * including its parents (input nodes), the operator that produced it, and
 * the logic for backpropagation (gradFun).
 *
 * In a dynamic computational graph:
 * - Nodes are created during the forward pass.
 * - Gradients are propagated during the backward pass using the chain rule:
 *   dL/dx = dL/dz * dz/dx
 *
 * @tparam T The numeric data type (e.g., float, double).
 */
template <typename T> class ComputeNode {
    static_assert(utils::isLegalDataType<T>::value,
                  "T must be a legal data type");

  public:
    /**
     * @brief Construct a leaf node (e.g., a constant or parameter).
     * @param data The tensor data managed by this node.
     */
    explicit ComputeNode(std::shared_ptr<math::TensorWrapper<T>> data)
        : data_(data) {
    }

    /**
     * @brief Construct a node with a single result tensor and an operation.
     *        Used for unary operations or when the graph structure is built
     * step-by-step.
     * @param res Result tensor data.
     * @param operatorType The operation performed.
     * @param gradFun The function to compute gradients.
     */
    ComputeNode(std::shared_ptr<math::TensorWrapper<T>> res,
                common::Operator operatorType,
                std::function<void()> gradFun = nullptr)
        : data_(res), operatorType_(operatorType),
          gradFun_(std::move(gradFun)) {
        if (operatorType_ == common::Operator::None) {
            throw std::invalid_argument("Operator cannot be None");
        }
    }

    /**
     * @brief Construct a node as a result of an operation.
     * @param lhs Left-hand side parent node.
     * @param rhs Right-hand side parent node.
     * @param res Result tensor data.
     * @param operatorType The operation performed.
     * @param gradFun The function to compute gradients for parents.
     */
    ComputeNode(std::shared_ptr<ComputeNode<T>> lhs,
                std::shared_ptr<ComputeNode<T>> rhs,
                std::shared_ptr<math::TensorWrapper<T>> res,
                common::Operator operatorType,
                std::function<void()> gradFun)
        : data_(res), operatorType_(operatorType),
          gradFun_(std::move(gradFun)) {
        if (operatorType_ == common::Operator::None) {
            throw std::invalid_argument("Operator cannot be None");
        }
        parents_.push_back(lhs);
        parents_.push_back(rhs);
        // Automatically determine if this node requires gradients based on
        // parents
        this->requiresGrad_ =
            lhs->requiresGrad_ || rhs->requiresGrad_ || this->requiresGrad_;
    }

    /**
     * @brief Accumulates incoming gradients into this node's gradient tensor.
     *
     * In the backward pass, a node might receive gradients from multiple
     * children (e.g., if it's used multiple times in an expression).
     * Formula: grad_total = sum(incoming_gradients)
     *
     * @param grad The incoming gradient tensor.
     */
    void accumulateGrad(std::shared_ptr<math::TensorWrapper<T>> grad) {
        if (this->grad_) {
            // grad_total = grad_total + incoming_grad
            this->grad_ = std::make_shared<math::TensorWrapper<T>>(
                this->grad_->add(*grad));
        } else {
            // First gradient received, clone it to avoid side effects
            this->grad_ = std::make_shared<math::TensorWrapper<T>>(*grad);
        }
    }

    /**
     * @brief Add a parent node to this node's dependency list.
     * @param node The parent node.
     */
    void addParent(std::shared_ptr<ComputeNode<T>> node) {
        this->parents_.push_back(node);
    }

    /**
     * @brief Get the managed tensor data.
     * @return shared_ptr to the data.
     */
    std::shared_ptr<math::TensorWrapper<T>> getData() {
        return this->data_;
    }

    /**
     * @brief Set the gradient tensor for this node.
     * @param grad The gradient tensor.
     */
    void setGrad(std::shared_ptr<math::TensorWrapper<T>> grad) {
        this->grad_ = grad;
    }

    /**
     * @brief Get the current gradient tensor.
     * @return shared_ptr to the gradient.
     */
    std::shared_ptr<math::TensorWrapper<T>> getGrad() {
        return this->grad_;
    }

    /**
     * @brief Clean the gradient of the node.
     */
    void clearGrad() {
        grad_->clear();
        for (auto& parent : parents_) {
            if (parent) {
                parent->clearGrad();
            }
        }
    }

    /**
     * @brief Check if this node requires gradient calculation.
     * @return true if gradients are needed.
     */
    [[nodiscard]] bool getRequiresGrad() const {
        return requiresGrad_;
    }

    /**
     * @brief Set whether this node requires gradients.
     * @param req Boolean flag.
     */
    void setRequiresGrad(bool req) {
        this->requiresGrad_ = req;
    }

    /**
     * @brief Set the gradient function for backpropagation.
     * @param gradFun The function to compute gradients.
     */
    void setGradFun(std::function<void()> gradFun) {
        gradFun_ = std::move(gradFun);
    }

    /** @brief Get the gradient function. */
    std::function<void()> getGradFun() {
        return gradFun_;
    }

    /**
     * @brief Triggers the backward pass starting from this node.
     *
     * If this is the output node (e.g., Loss), its gradient is initialized
     * to 1.0 (dL/dL = 1). Then it calls its gradFun to propagate gradients
     * to parents.
     */
    void backward() {
        if (!requiresGrad_) {
            return;
        }
        // If it's the root of the backward pass (Loss), initialize dL/dz = 1
        if (!grad_) {
            grad_ = std::make_shared<math::TensorWrapper<T>>(
                math::TensorShape(data_->getShape()), 1);
        }
        // Call the operator-specific gradient function
        if (gradFun_) {
            gradFun_();
        }
    }

    static std::shared_ptr<ComputeNode>
    createUnary(std::shared_ptr<ComputeNode> parent,
                std::shared_ptr<math::TensorWrapper<T>> res,
                common::Operator operatorType,
                std::function<void()> gradFun = nullptr) {
        std::shared_ptr<ComputeNode> node = std::make_shared<ComputeNode>(
            res, operatorType, std::move(gradFun));
        node->addParent(parent);
        node->setRequiresGrad(parent->getRequiresGrad());
        return node;
    }

  private:
    std::vector<std::shared_ptr<ComputeNode>> parents_; /**< Input nodes. */
    std::shared_ptr<math::TensorWrapper<T>> data_;         /**< Forward data. */
    common::Operator operatorType_ =
        common::Operator::None; /**< Operation used. */

    bool requiresGrad_ = false;     /**< Grad requirement flag. */
    std::function<void()> gradFun_; /**< Backprop logic. */
    std::shared_ptr<math::TensorWrapper<T>> grad_; /**< Accumulated grad. */

    friend class hahaha::Tensor<T>;
    template <typename U> friend class TopoSort;
};

} // namespace hahaha::compute

#endif // HAHAHA_COMPUTE_COMPUTE_GRAPH_COMPUTE_NODE_H
