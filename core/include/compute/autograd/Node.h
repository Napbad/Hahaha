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

#ifndef HAHAHA_COMPUTE_AUTOGRAD_NODE_H
#define HAHAHA_COMPUTE_AUTOGRAD_NODE_H

#include <memory>
#include <vector>

#include "math/Tensor.h"

namespace hahaha::autograd
{

template <typename T> class Function;

/**
 * @brief Base class for all nodes in the computation graph.
 *
 * A node represents an operation or a variable in the graph.
 * It tracks its inputs (predecessors) and the function that created it.
 *
 * @tparam T The numeric type of the node's data.
 */
template <typename T> class Node
{
  public:
    Node() = default;
    virtual ~Node() = default;

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    Node(Node&&) noexcept = default;
    Node& operator=(Node&&) noexcept = default;

    /**
     * @brief Get the inputs to this node.
     * @return const std::vector<std::shared_ptr<Node<T>>>& reference to inputs.
     */
    const std::vector<std::shared_ptr<Node<T>>>& inputs() const { return inputs_; }

    /**
     * @brief Add an input to this node.
     * @param input The input node to add.
     */
    void addInput(std::shared_ptr<Node<T>> input) { inputs_.push_back(std::move(input)); }

    /**
     * @brief Get the function that created this node.
     * @return std::shared_ptr<Function<T>> the creator function.
     */
    std::shared_ptr<Function<T>> creator() const { return creator_; }

    /**
     * @brief Set the function that created this node.
     * @param creator The creator function.
     */
    void setCreator(std::shared_ptr<Function<T>> creator) { creator_ = std::move(creator); }

    /**
     * @brief Get the gradient stored in this node.
     * @return std::shared_ptr<hahaha::math::Tensor<T>> the gradient.
     */
    std::shared_ptr<hahaha::math::Tensor<T>> grad() const { return grad_; }

    /**
     * @brief Set the gradient for this node.
     * @param grad The gradient tensor.
     */
    void setGrad(std::shared_ptr<hahaha::math::Tensor<T>> grad) { grad_ = std::move(grad); }

    /**
     * @brief Accumulate gradient into this node.
     * @param grad The gradient to add.
     */
    void addGrad(const std::shared_ptr<hahaha::math::Tensor<T>>& grad);

  private:
    std::vector<std::shared_ptr<Node<T>>> inputs_;
    std::shared_ptr<Function<T>> creator_;
    std::shared_ptr<hahaha::math::Tensor<T>> grad_;
};

} // namespace hahaha::autograd

#endif // HAHAHA_COMPUTE_AUTOGRAD_NODE_H
