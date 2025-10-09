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
// Created by napbad on 9/28/25.
//

#ifndef HAHAHA_COMPUTEGRAPHNODE_H
#define HAHAHA_COMPUTEGRAPHNODE_H
#include <functional>

#include "ml/common/Tensor.h"

namespace hahaha::ml::util
{
/**
 * ComputeGraphNode class
 *
 * Represents a node in a computational graph
 */

template <typename T> class ComputeGraphNode
{

  public:
    ComputeGraphNode() = default;
    ~ComputeGraphNode() = default;

  private:
    Tensor<T> tensor_;
    Tensor<T> grad_;
    ds::Vector<ComputeGraphNode*> inputs_;
    ds::Vector<ComputeGraphNode*> outputs_;
    std::function<Tensor<T>(ds::Vector<Tensor<T>>)> forwardFn_;
    std::function<ds::Vector<Tensor<T>>(Tensor<T>)> backwardFn_;

    Tensor<T> forwardFnAdd(ds::Vector<Tensor<T>> vals)
    {
        grad_ = Tensor<T>(vals.begin()->shape()).fill(1);
        // create a Tensor from specified shape
        Tensor<T> res = Tensor<T>(vals.begin()->shape());
        res.fill(0);
        for (auto& val : vals)
        {
        }
        return res;
    }

    Tensor<T> forwardRnSub(ds::Vector<Tensor<T>> vals)
    {
        grad_ = Tensor<T>::fill(1);
        // create a Tensor from specified shape
        Tensor<T> res = Tensor<T>(vals.begin()->shape());
        res.copy(vals.begin);
        for (auto& val : vals)
        {
        }
    }
};

} // namespace hahaha::ml::util

#endif // HAHAHA_COMPUTEGRAPHNODE_H
