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
#include "TensorData.h"

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

using ml::TensorData;

// one tensor can only be hold by 1 TensorVar and 1 GraphNode
// this class will act as a wrapper for Tensor like shared_ptr
template <typename T> class TensorPtr
{
    friend class TensorVar<T>;
    friend class ad::ComputeNode<T>;

  public:
    ~TensorPtr() = default;

    TensorPtr(const std::initializer_list<sizeT> shape)
    {
        refcount_ = new i8(0);
        tensor_ = new TensorData<T>(shape);
    }

    explicit TensorPtr(const ds::Vector<sizeT>& shape)
    {
        refcount_ = new i8(0);
        tensor_ = new TensorData<T>(shape);
    }
    explicit TensorPtr(const ds::Vector<sizeT>& shape, const T* data)
    {
        refcount_ = new i8(0);
        tensor_ = new TensorData<T>(shape, data);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(T scalar)
    {
        refcount_ = new i8(0);
        tensor_ = new TensorData<T>(scalar);
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorPtr(const std::initializer_list<sizeT> shape,
                       std::initializer_list<T> data)
    {
        refcount_ = new i8(0);
        tensor_ = new TensorData<T>(shape, data);
    }

    void refByVar() const
    {
        if ((TENSOR_VAR_REF_MASK & *refcount_) != 0)
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

    void unrefByVar() const
    {
        if (refcount_ == nullptr)
        {
            return;
        }
        if ((TENSOR_VAR_REF_MASK & *refcount_) == 0)
        {
            throw std::runtime_error("Var bit not set");
        }
        *refcount_ = static_cast<i8>(*refcount_ & ~TENSOR_VAR_REF_MASK);
        if (*refcount_ == 0)
        {
            delete refcount_;
            delete tensor_;
            const_cast<TensorPtr<T>*>(this)->refcount_ = nullptr;
            const_cast<TensorPtr<T>*>(this)->tensor_ = nullptr;
        }
    }

    void unrefByNode() const
    {
        if (refcount_ == nullptr)
        {
            return;
        }
        if ((COMPUTE_NODE_REF_MASK & *refcount_) == 0)
        {
            throw std::runtime_error("Node bit not set");
        }
        *refcount_ = static_cast<i8>(*refcount_ & ~COMPUTE_NODE_REF_MASK);
        if (*refcount_ == 0)
        {
            delete refcount_;
            delete tensor_;
            const_cast<TensorPtr<T>*>(this)->refcount_ = nullptr;
            const_cast<TensorPtr<T>*>(this)->tensor_ = nullptr;
        }
    }

    TensorPtr(const TensorPtr& other) = delete;
    TensorPtr(TensorPtr&& other) noexcept
    {
        this->tensor_ = other.tensor_;
        this->refcount_ = other.refcount_;
        other.tensor_ = nullptr;
        other.refcount_ = nullptr;
    }
    TensorPtr& operator=(const TensorPtr& other) = delete;
    TensorPtr& operator=(TensorPtr&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        // steal
        this->tensor_ = other.tensor_;
        this->refcount_ = other.refcount_;
        other.tensor_ = nullptr;
        other.refcount_ = nullptr;
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

    explicit TensorPtr (TensorData<T>&& tensor)
    {
        this->tensor_ = new TensorData<T>(std::move(tensor));
        refcount_ = new i8(0);
    }

    explicit TensorPtr(TensorData<T>& tensor)
    {
        this->tensor_ = new TensorData<T>(tensor);
        refcount_ = new i8(0);
    }

    explicit TensorPtr(TensorData<T> *tensor )
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

    void releaseOwnership_() = delete;

    TensorData<T>* tensor_;
    i8* refcount_;
};
} // namespace hahaha::core

#endif // HAHAHA_TENSORPTR_H
