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
template <typename T> class ComputeNode;
}

namespace hahaha
{
template <typename T> class TensorVar;
}

namespace hahaha::core
{

using ml::Tensor;

// one tensor can only be hold by 1 TensorVar and 1 GraphNode
// this class will act as a wrapper for Tensor like shared_ptr
template <typename T> class TensorPtr
{
    friend class TensorVar<T>;
    friend class ad::ComputeNode<T>;

  public:
    ~TensorPtr()
    {
        *refcount_ -= 1;
        if (*refcount_ == 0)
        {
            delete refcount_;
            delete tensor_;
        }
    }

    TensorPtr(const std::initializer_list<sizeT> shape)
    {
        refcount_ = new i8(0);
        tensor_ = new Tensor<T>(shape);
    }

    explicit TensorPtr(const ds::Vector<sizeT>& shape)
    {
        refcount_ = new i8(0);
        tensor_ = new Tensor<T>(shape);
    }
    explicit TensorPtr(const ds::Vector<sizeT>& shape, const T* data)
    {
        refcount_ = new i8(0);
        tensor_ = new Tensor<T>(shape, data);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(T scalar)
    {
        refcount_ = new i8(0);
        tensor_ = new Tensor<T>(scalar);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(const std::initializer_list<sizeT> shape,
                       std::initializer_list<T> data)
    {
        refcount_ = new i8(0);
        tensor_ = new Tensor<T>(shape, data);
    }

    void refByVar() const
    {
        if (*refcount_ > TENSOR_VAR_REF_MASK)
            throw std::runtime_error(
                "TensorPtr already referenced by a TensorVar");
        *refcount_ |= TENSOR_VAR_REF_MASK;
    }

    void refByNode() const
    {
        if ((COMPUTE_NODE_REF_MASK & *refcount_) != 0)
            throw std::runtime_error(
                "TensorPtr already referenced by a ComputeNode");
        *refcount_ |= COMPUTE_NODE_REF_MASK;
    }

    TensorPtr(const TensorPtr& other) = delete;
    TensorPtr(TensorPtr&& other) = delete;
    TensorPtr& operator=(const TensorPtr& other) = delete;
    TensorPtr& operator=(TensorPtr&& other) noexcept
    {
        *this = std::move(other);
        return *this;
    };

    explicit operator bool() const
    {
        return tensor_ != nullptr;
    }

  private:
    static constexpr i8 TENSOR_VAR_REF_MASK = 0b00000010;
    static constexpr i8 COMPUTE_NODE_REF_MASK = 0b00000001;

    TensorPtr()
    {
        refcount_ = new i8(0);
        tensor_ = nullptr;
    }

    explicit TensorPtr (Tensor<T>&& tensor)
    {
        this->tensor_ = new Tensor<T>(std::move(tensor));
        refcount_ = new i8(0);
    }

    explicit TensorPtr(Tensor<T>& tensor)
    {
        this->tensor_ = new Tensor<T>(tensor);
        refcount_ = new i8(0);
    }

    explicit TensorPtr(Tensor<T> *tensor )
    {
        this->tensor_ = tensor;
        refcount_ = new i8(0);
    }

    // only TensorVar and ComputeNode can initialize the TensorPtr
    [[nodiscard]] bool holdByVar() const
    {
        return TENSOR_VAR_REF_MASK & *refcount_;
    }
    [[nodiscard]] bool holdByNode() const
    {
        return COMPUTE_NODE_REF_MASK & *refcount_;
    }

    Tensor<T>* tensor_;
    i8* refcount_;
};
} // namespace hahaha::core

#endif // HAHAHA_TENSORPTR_H
