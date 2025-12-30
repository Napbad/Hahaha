// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
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

#ifndef HAHAHA_TENSOR_H
#define HAHAHA_TENSOR_H

#include <memory>
#include <type_traits>

#include "compute/compute_graph/ComputeFun.h"
#include "math/ds/TensorData.h"
#include "utils/common/HelperStruct.h"

namespace hahaha {

/**
 * @brief High-level User Interface for Tensor operations and Autograd.
 *
 * This class acts as a handle to a ComputeNode in the computational graph.
 * It provides operator overloading (+, -, *, /) which automatically builds
 * the graph in the background (Dynamic Graph / Define-by-Run).
 *
 * Usage:
 *   Tensor<float> a({2, 2}, 1.0f);
 *   Tensor<float> b({2, 2}, 2.0f);
 *   auto c = a + b;
 *   c.backward(); // Propagates gradients back to a and b
 *
 * @tparam T Numeric data type.
 */
template <typename T> class Tensor {
    static_assert(utils::isLegalDataType<T>::value,
                  "T must be a legal data type");

  public:
    /**
     * @brief Construct a Tensor from an existing TensorWrapper.
     * @param data The numerical data wrapper.
     */
    // NOLINTNEXTLINE
    Tensor(const math::TensorWrapper<T>& data)
        : computeNode_(std::make_shared<compute::ComputeNode<T>>(
              std::make_shared<math::TensorWrapper<T>>(data))) {}

    /**
     * @brief Construct a Tensor from a shared pointer to TensorWrapper.
     * @param dataPtr pointer to the numerical data.
     */
    // NOLINTNEXTLINE
    Tensor(std::shared_ptr<math::TensorWrapper<T>> dataPtr)
        : computeNode_(std::make_shared<compute::ComputeNode<T>>(dataPtr)) {}

    /**
     * @brief Internal constructor to wrap a ComputeNode.
     * @param computeNode The node in the computational graph.
     */
    explicit Tensor(std::shared_ptr<compute::ComputeNode<T>> computeNode)
        : computeNode_(computeNode) {}

    /** @brief Addition operator. Builds an 'Add' node. */
    Tensor<T> operator+(const Tensor<T>& other) const {
        return Tensor(compute::add(this->computeNode_, other.computeNode_));
    }

    /** @brief Subtraction operator. Builds a 'Sub' node. */
    Tensor<T> operator-(const Tensor<T>& other) const {
        return Tensor(compute::sub(this->computeNode_, other.computeNode_));
    }

    /** @brief Multiplication operator. Builds a 'Mul' node. */
    Tensor<T> operator*(const Tensor<T>& other) const {
        return Tensor(compute::mul(this->computeNode_, other.computeNode_));
    }

    /** @brief Division operator. Builds a 'Div' node. */
    Tensor<T> operator/(const Tensor<T>& other) const {
        return Tensor(compute::div(this->computeNode_, other.computeNode_));
    }

    /**
     * @brief Triggers backpropagation from this tensor.
     *
     * This will compute gradients for all ancestor tensors in the graph
     * that have 'requiresGrad' set to true.
     */
    void backward() { computeNode_->backward(); }

    /**
     * @brief Get the underlying compute node.
     * @return shared_ptr to the node.
     */
    std::shared_ptr<compute::ComputeNode<T>> getComputeNode() {
        return computeNode_;
    }

    /**
     * @brief Set the compute node for this tensor.
     * @param node The new node.
     */
    void setComputeNode(std::shared_ptr<compute::ComputeNode<T>> node) {
        computeNode_ = node;
    }

  private:
    std::shared_ptr<compute::ComputeNode<T>> computeNode_; /**< Graph link. */
};

} // namespace hahaha

#endif // HAHAHA_TENSOR_H
