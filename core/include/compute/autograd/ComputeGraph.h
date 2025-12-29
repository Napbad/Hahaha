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

#ifndef HAHAHA_COMPUTE_AUTOGRAD_COMPUTE_GRAPH_H
#define HAHAHA_COMPUTE_AUTOGRAD_COMPUTE_GRAPH_H

#include <memory>
#include <vector>

#include "compute/autograd/Node.h"

namespace hahaha::autograd
{

/**
 * @brief Manages the structure and execution of the computation graph.
 *
 * ComputeGraph tracks all nodes and facilitates operations like
 * topological sorting and execution.
 *
 * @tparam T The numeric type of the graph's data.
 */
template <typename T> class ComputeGraph
{
  public:
    ComputeGraph() = default;
    ~ComputeGraph() = default;

    ComputeGraph(const ComputeGraph&) = delete;
    ComputeGraph& operator=(const ComputeGraph&) = delete;

    ComputeGraph(ComputeGraph&&) noexcept = default;
    ComputeGraph& operator=(ComputeGraph&&) noexcept = default;

    /**
     * @brief Add a node to the graph.
     * @param node The node to add.
     */
    void addNode(std::shared_ptr<Node<T>> node)
    {
        nodes_.push_back(std::move(node));
    }

    /**
     * @brief Clear all nodes from the graph.
     */
    void clear()
    {
        nodes_.clear();
    }

    /**
     * @brief Get all nodes in the graph.
     * @return const std::vector<std::shared_ptr<Node<T>>>& reference to nodes.
     */
    const std::vector<std::shared_ptr<Node<T>>>& nodes() const
    {
        return nodes_;
    }

  private:
    std::vector<std::shared_ptr<Node<T>>> nodes_;
};

} // namespace hahaha::autograd

#endif // HAHAHA_COMPUTE_AUTOGRAD_COMPUTE_GRAPH_H
