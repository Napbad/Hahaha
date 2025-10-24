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

#ifndef HAHAHA_TENSORVAR_H
#define HAHAHA_TENSORVAR_H

#include "core/compute/autodiff/GraphNode.h"
#include "core/ds/String.h"
#include "TensorPtr.h"
#include <stdexcept>

namespace hahaha
{

using core::ds::String;
using core::ds::Vector;
using core::TensorPtr;

template <typename T> class TensorVar
{
  public:
    explicit TensorVar(String name = String()) : ptr_(nullptr), name_(std::move(name))
    {
    }

    TensorVar(const std::initializer_list<sizeT> shape, String name = String())
        : ptr_(new TensorPtr<T>(shape)), name_(std::move(name))
    {
        ptr_->refByVar();
    }

    explicit TensorVar(const Vector<sizeT>& shape, String name = String())
        : ptr_(new TensorPtr<T>(shape)), name_(std::move(name))
    {
        ptr_->refByVar();
    }
    explicit TensorVar(const Vector<sizeT>& shape,
                       const T* data,
                       String name = String())
        : ptr_(new TensorPtr<T>(shape, data)), name_(std::move(name))
    {
        ptr_->refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(T scalar, String name = String())
        : ptr_(new TensorPtr<T>(scalar)), name_(std::move(name))
    {
        ptr_->refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(const std::initializer_list<sizeT> shape,
                       std::initializer_list<T> data,
                       String name = String())
        : ptr_(new TensorPtr<T>(shape, data)), name_(std::move(name))
    {
        ptr_->refByVar();
    }

    // ===================== Tensor-like API exposure =====================
    // Basic accessors mirroring Tensor
    [[nodiscard]] const Vector<sizeT>& shape() const { return getTensor_().shape(); }
    [[nodiscard]] sizeT size() const { return getTensor_().size(); }
    T operator[](sizeT idx) const { return getTensor_()[idx]; }

    // Access underlying Tensor reference
    ml::Tensor<T>& tensor() { return getTensor_(); }
    const ml::Tensor<T>& tensor() const { return getTensor_(); }

    // Data view (const, matches Tensor<T>::data API)
    [[nodiscard]] const Vector<T>& data() const { return getTensor_().data(); }

    // Conversions to Tensor reference (explicit to avoid accidental decay)
    explicit operator ml::Tensor<T>&() { return getTensor_(); }
    explicit operator const ml::Tensor<T>&() const { return getTensor_(); }

    // Pointer-like forwarding
    ml::Tensor<T>* operator->() { return &getTensor_(); }
    const ml::Tensor<T>* operator->() const { return &getTensor_(); }
    ml::Tensor<T>& operator*() { return getTensor_(); }
    const ml::Tensor<T>& operator*() const { return getTensor_(); }

    TensorVar& operator=(TensorVar&& other) noexcept {
        this->ptr_ = other.ptr_;
        return *this;
    }

  private:
    // Safety: centralize null-check
    ml::Tensor<T>& getTensor_() const
    {
        if (!ptr_)
            throw std::runtime_error("TensorVar is empty (null TensorPtr)");
        if (ptr_->tensor_ == nullptr)
            throw std::runtime_error("TensorVar holds null Tensor pointer");
        return ptr_->tensor_;
    }

    TensorPtr<T> ptr_;
    String name_;
};

} // namespace hahaha

#endif // HAHAHA_TENSORVAR_H
