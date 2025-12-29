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

#ifndef HAHAHA_COMPUTE_AUTOGRAD_VARIABLE_H
#define HAHAHA_COMPUTE_AUTOGRAD_VARIABLE_H

#include <memory>

#include "compute/autograd/Node.h"
#include "math/Tensor.h"

namespace hahaha::autograd
{

/**
 * @brief A Variable wraps a Tensor and its corresponding Node in the computation graph.
 *
 * Variables are the primary objects users interact with in the autograd system.
 * They maintain a pointer to the underlying data (Tensor) and a node that
 * tracks dependencies for backpropagation.
 *
 * @tparam T The numeric type of the variable's data.
 */
template <typename T> class Variable
{
  public:
    /**
     * @brief Construct a Variable from a Tensor.
     * @param tensor The data for the variable.
     * @param requires_grad Whether this variable needs to track gradients.
     */
    explicit Variable(std::shared_ptr<hahaha::math::Tensor<T>> tensor, bool requires_grad = false)
        : data_(std::move(tensor)), requires_grad_(requires_grad)
    {
        node_ = std::make_shared<Node<T>>();
    }

    Variable(const Variable&) = delete;
    Variable& operator=(const Variable&) = delete;

    Variable(Variable&& other) noexcept = default;
    Variable& operator=(Variable&& other) noexcept = default;

    /**
     * @brief Get the underlying tensor data.
     * @return std::shared_ptr<hahaha::math::Tensor<T>> the tensor data.
     */
    std::shared_ptr<hahaha::math::Tensor<T>> data() const { return data_; }

    /**
     * @brief Get the gradient tensor for this variable.
     * @return std::shared_ptr<hahaha::math::Tensor<T>> the gradient tensor.
     */
    std::shared_ptr<hahaha::math::Tensor<T>> grad() const { return node_->grad(); }

    /**
     * @brief Set the gradient tensor for this variable.
     * @param grad The gradient tensor.
     */
    void setGrad(std::shared_ptr<hahaha::math::Tensor<T>> grad) { node_->setGrad(std::move(grad)); }

    /**
     * @brief Get the node associated with this variable in the computation graph.
     * @return std::shared_ptr<Node<T>> the node.
     */
    std::shared_ptr<Node<T>> node() const { return node_; }

    /**
     * @brief Check if this variable requires gradient tracking.
     * @return bool true if it requires grad.
     */
    bool requiresGrad() const { return requires_grad_; }

    /**
     * @brief Perform backpropagation starting from this variable.
     */
    void backward();

  private:
    std::shared_ptr<hahaha::math::Tensor<T>> data_;
    std::shared_ptr<Node<T>> node_;
    bool requires_grad_;
};

} // namespace hahaha::autograd

#endif // HAHAHA_COMPUTE_AUTOGRAD_VARIABLE_H
