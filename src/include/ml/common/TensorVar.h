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

#include <stdexcept>
#include <memory>
#include <utility>

#include "TensorPtr.h"
#include "TensorVarOp.h"
#include "core/ds/String.h"

namespace hahaha
{

using core::ds::String;
using core::ds::Vector;
using core::TensorPtr;

template <typename T> class TensorVar;

template<typename T>
using TensorVarPtr = std::shared_ptr<TensorVar<T>>;

template <typename T> class TensorVar
{
  public:
    ~TensorVar()
    {
        // Ensure Variable bit is released when TensorVar goes out of scope
        if (ptr_ && ptr_.holdByVar())
            ptr_.unrefByVar();
    }

    TensorVar(const TensorVar& other) = delete;
    TensorVar& operator=(const TensorVar& other) = delete;

    TensorVar(TensorVar&& other) noexcept = default;
    TensorVar& operator=(TensorVar&& other) noexcept = default;

    TensorVar(const TensorVarOpType opType, std::initializer_list<TensorVarPtr<T>> operands)
    {
        opType_ = opType;
        for (auto& operand : operands)
            parents_.pushBack(operand);
    }

    explicit TensorVar(String name = String()) : ptr_(nullptr), name_(std::move(name))
    {
    }

    TensorVar(const std::initializer_list<sizeT> shape, String name = String())
        : ptr_(TensorPtr<T>(shape)), name_(std::move(name))
    {
        ptr_.refByVar();
    }

    // Convenience overloads to accept C-string names
    TensorVar(const std::initializer_list<sizeT> shape, const char* name)
        : ptr_(TensorPtr<T>(shape)), name_(String(name))
    {
        ptr_.refByVar();
    }

    explicit TensorVar(const Vector<sizeT>& shape, String name = String())
        : ptr_(TensorPtr<T>(shape)), name_(std::move(name))
    {
        ptr_.refByVar();
    }

    explicit TensorVar(const Vector<sizeT>& shape, const char* name)
        : ptr_(TensorPtr<T>(shape)), name_(String(name))
    {
        ptr_.refByVar();
    }
    explicit TensorVar(const Vector<sizeT>& shape,
                       const T* data,
                       String name = String())
        : ptr_(TensorPtr<T>(shape, data)), name_(std::move(name))
    {
        ptr_.refByVar();
    }

    explicit TensorVar(const Vector<sizeT>& shape,
                       const T* data,
                       const char* name)
        : ptr_(TensorPtr<T>(shape, data)), name_(String(name))
    {
        ptr_.refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(T scalar, String name = String())
        : ptr_(TensorPtr<T>(scalar)), name_(std::move(name))
    {
        ptr_.refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(T scalar, const char* name)
        : ptr_(TensorPtr<T>(scalar)), name_(String(name))
    {
        ptr_.refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(const std::initializer_list<sizeT> shape,
                       std::initializer_list<T> data,
                       String name = String())
        : ptr_(TensorPtr<T>(shape, data)), name_(std::move(name))
    {
        ptr_.refByVar();
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit TensorVar(const std::initializer_list<sizeT> shape,
                       std::initializer_list<T> data,
                       const char* name)
        : ptr_(TensorPtr<T>(shape, data)), name_(String(name))
    {
        ptr_.refByVar();
    }

    TensorVarPtr<T> calc();

    // ===================== Tensor-like API exposure =====================
    // Basic accessors mirroring Tensor
    [[nodiscard]] const Vector<sizeT>& shape() const { return getTensorData_().shape(); }
    [[nodiscard]] sizeT size() const { return getTensorData_().size(); }
    T operator[](sizeT idx) const { return getTensorData_()[idx]; }

    // Access underlying Tensor reference
    ml::TensorData<T>& tensorData() { return getTensorData_(); }
    const ml::TensorData<T>& tensorData() const { return getTensorData_(); }

    // Data view (const, matches Tensor<T>::data API)
    [[nodiscard]] const Vector<T>& data() const { return getTensorData_().data(); }

    // Conversions to Tensor reference (explicit to avoid accidental decay)
    explicit operator ml::TensorData<T>&() { return getTensorData_(); }
    explicit operator const ml::TensorData<T>&() const { return getTensorData_(); }

    // Pointer-like forwarding
    ml::TensorData<T>* operator->() { return &getTensorData_(); }
    const ml::TensorData<T>* operator->() const { return &getTensorData_(); }
    ml::TensorData<T>& operator*() { return getTensorData_(); }
    const ml::TensorData<T>& operator*() const { return getTensorData_(); }

    // moved versions are defaulted above

  private:
    // Safety: centralize null-check
    ml::TensorData<T>& getTensorData_()
    {
        if (!ptr_)
            throw std::runtime_error("TensorVar is empty (null TensorPtr)");
        if (ptr_.tensor_ == nullptr)
            throw std::runtime_error("TensorVar holds null Tensor pointer");
        return *ptr_.tensor_;
    }

    const ml::TensorData<T>& getTensorData_() const
    {
        if (!ptr_)
            throw std::runtime_error("TensorVar is empty (null TensorPtr)");
        if (ptr_.tensor_ == nullptr)
            throw std::runtime_error("TensorVar holds null Tensor pointer");
        return *ptr_.tensor_;
    }

    TensorPtr<T> ptr_;
    String name_;
    Vector<TensorVarPtr<T>> parents_;

    TensorVarOpType opType_ = TensorVarOpType::None;
};

template<typename T>
using Tensor = TensorVarPtr<T>;

using Tensori = TensorVar<i32>;
using Tensorf = TensorVar<f32>;
using Tensorb = TensorVar<bool>;
using Tensoru = TensorVar<u32>;
using Tensorl = TensorVar<i64>;
using Tensorul = TensorVar<u64>;
using Tensord = TensorVar<f64>;
using Tensorc = TensorVar<char>;

// Factory helper to construct a TensorVarPtr with perfect forwarding.
// This is a convenience layer over TensorVar constructors.
template <typename T, typename... Args>
Tensor<T> tensor(Args&&... args)
{
    return std::make_shared<TensorVar<T>>(std::forward<Args>(args)...);
}

template<typename T>
Tensor<T> operator+ (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return std::make_shared<TensorVar<T>>(TensorVarOpType::Add, {lhs, rhs});
}

template<typename T>
Tensor<T> operator- (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return std::make_shared<TensorVar<T>>(TensorVarOpType::Sub, {lhs, rhs});
}

template<typename T>
Tensor<T> operator* (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return std::make_shared<TensorVar<T>>(TensorVarOpType::Mul, {lhs, rhs});
}

template<typename T>
Tensor<T> operator/ (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return std::make_shared<TensorVar<T>>(TensorVarOpType::Div, {lhs, rhs});
}



} // namespace hahaha

#endif // HAHAHA_TENSORVAR_H
