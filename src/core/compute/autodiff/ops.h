// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

#ifndef HAHAHA_OPS_H
#define HAHAHA_OPS_H

#include <memory>

#include "ComputeNode.h"
#include "GraphNode.h"

namespace hahaha::ad
{

// Represents the addition of two nodes in the graph.
template <typename T> class AddNode : public ComputeNode<T>
{
  public:
    AddNode(std::shared_ptr<GraphNode<T>> lhs,
            std::shared_ptr<GraphNode<T>> rhs)
        : ComputeNode<T>({lhs, rhs})
    {
    }

    void forward() override
    {
        // In a static graph, the actual computation is handled by an executor
        // after the graph is compiled. This function would be called by the
        // executor.
    }
};

// Represents the multiplication of two nodes.
template <typename T> class MultiplyNode : public ComputeNode<T>
{
  public:
    MultiplyNode(std::shared_ptr<GraphNode<T>> lhs,
                 std::shared_ptr<GraphNode<T>> rhs)
        : ComputeNode<T>({lhs, rhs})
    {
    }

    void forward() override
    {
        // Logic handled by graph executor.
    }
};

template <typename T> class SubtractNode : public ComputeNode<T>
{
public:
    SubtractNode(std::shared_ptr<GraphNode<T>> lhs,
                std::shared_ptr<GraphNode<T>> rhs
    ) : ComputeNode<T>({lhs, rhs})
    {
    }

    void forward() override
    {

    }
};

template <typename T> class DivideNode : public ComputeNode<T>
{
public:
    DivideNode(std::shared_ptr<GraphNode<T>> lhs,
              std::shared_ptr<GraphNode<T>> rhs
    ): ComputeNode<T>({lhs, rhs})
    {
    }

    void forward() override
    {

    }
};

// --- Operator Overloads ---
template <typename T>
GraphNodePtr<T>
operator+(const GraphNodePtr<T>& lhs,
          const GraphNodePtr<T>& rhs)
{
    return std::make_shared<AddNode<T>>(lhs, rhs);
}

template <typename T>
GraphNodePtr<T>
operator*(const GraphNodePtr<T>& lhs,
          const GraphNodePtr<T>& rhs)
{
    return std::make_shared<MultiplyNode<T>>(lhs, rhs);
}

template <typename T>
GraphNodePtr<T>
operator-(const GraphNodePtr<T>& lhs,
        const GraphNodePtr<T>& rhs)
{
    return std::make_shared<SubtractNode<T>>(lhs, rhs);
}

template <typename T>
GraphNodePtr<T> operator/(const GraphNodePtr<T>& lhs,
    const GraphNodePtr<T>& rhs)
{
    return std::make_shared<DivideNode<T>>(lhs, rhs);
}


} // namespace hahaha::ad

#endif // HAHAHA_OPS_H
