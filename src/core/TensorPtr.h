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
// Created by root on 10/22/25.
//

#ifndef HAHAHA_TENSORPTR_H
#define HAHAHA_TENSORPTR_H
#include "Tensor.h"

namespace hahaha::ad
{
template<typename T>
class ComputeNode;
}

namespace hahaha
{
template<typename T>
class TensorVar;
}

namespace hahaha::core
{

using ml::Tensor;

// one tensor can only be hold by 1 TensorVar and 1 GraphNode
// this class will act as a wrapper for Tensor like shared_ptr
template<typename T>
class TensorPtr
{
    friend class hahaha::TensorVar<T>;
    friend class hahaha::ad::ComputeNode<T>;

public:
    TensorPtr() = delete;

    explicit TensorPtr(const std::initializer_list<sizeT> shape) : refcount(0)
    {
        tensor_ = new Tensor<T>(shape);
    }

    explicit TensorPtr(const ds::Vector<sizeT>& shape) : refcount(0)
    {
        tensor_ = new Tensor<T> (shape);
    }
    explicit TensorPtr(const ds::Vector<sizeT>& shape, const T* data)
        :refcount(0)
    {
        tensor_ = new Tensor<T>(shape, data);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(T scalar) : refcount(0)
    {
        tensor_ = new Tensor<T>(scalar);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(const std::initializer_list<sizeT> shape,
                    std::initializer_list<T> data)
        : refcount(0)
    {
        tensor_ = new Tensor<T>(shape, data);
    }


    void refByVar()
    {
        if (refcount > TENSOR_VAR_REF_MASK)
            throw std::runtime_error("TensorPtr already referenced by a TensorVar");
        refcount |= TENSOR_VAR_REF_MASK;
    }

    void refByNode()
    {
        if ((COMPUTE_NODE_REF_MASK & refcount) != 0)
            throw std::runtime_error("TensorPtr already referenced by a ComputeNode");
        refcount |= COMPUTE_NODE_REF_MASK;
    }


    TensorPtr(const TensorPtr& other) = delete;
    TensorPtr(TensorPtr&& other) = delete;
    TensorPtr& operator=(const TensorPtr& other) = delete;
    TensorPtr& operator=(TensorPtr&& other) = delete;

private:
    static constexpr i8 TENSOR_VAR_REF_MASK = 0b00000010;
    static constexpr i8 COMPUTE_NODE_REF_MASK = 0b00000001;

    // only TensorVar and ComputeNode can initialize the TensorPtr
    

    [[nodiscard]] bool holdByVar() const
    {
        return TENSOR_VAR_REF_MASK & refcount;
    }
    [[nodiscard]] bool holdByNode() const
    {
        return COMPUTE_NODE_REF_MASK & refcount;
    }

    

    Tensor<T>* tensor_;
    i8 refcount;

};
}

#endif //HAHAHA_TENSORPTR_H
