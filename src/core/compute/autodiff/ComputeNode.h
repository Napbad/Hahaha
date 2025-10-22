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

//
// Created by root on 10/21/25.
//

#ifndef HAHAHA_NODE_H
#define HAHAHA_NODE_H
#include "Tensor.h"
#include "TensorPtr.h"
#include "GraphNode.h"
#include <vector>
#include <memory>

namespace hahaha::ad
{

using core::ds::Vector;
using core::Tensor;
using core::TensorPtr;

template<typename T>
class ComputeNode : public GraphNode<T>
{
  public:
    virtual ~ComputeNode() = default;

    ComputeNode() = default;

    explicit ComputeNode(std::vector<std::shared_ptr<GraphNode<T>>> srcs) : srcs_(std::move(srcs)) {}

    virtual void forward() = 0;
    // We will define backward logic later

    const std::vector<std::shared_ptr<GraphNode<T>>>& srcs() const { return srcs_; }
    Tensor<T>& grad() { return grad_; }

protected:
    std::vector<std::shared_ptr<GraphNode<T>>> srcs_;
    Tensor<T> grad_;
};
}

#endif // HAHAHA_NODE_H
