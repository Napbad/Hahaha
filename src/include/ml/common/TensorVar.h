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
#include <type_traits>

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

// Lightweight handle that wraps shared_ptr<TensorVar<T>> and forwards ergonomics
template <typename T> class Tensor
{
  public:
    using Element = TensorVar<T>;
    Tensor() = default;
    explicit Tensor(std::nullptr_t) : ptr_(nullptr) {}
    explicit Tensor(TensorVarPtr<T> p) : ptr_(std::move(p)) {}

    // Factory-friendly implicit conversion to shared_ptr for interop
    explicit operator TensorVarPtr<T>() const { return ptr_; }

    // Accessors
    TensorVar<T>* operator->() const { return ptr_.get(); }
    TensorVar<T>& operator*() const { return *ptr_; }
    explicit operator bool() const { return static_cast<bool>(ptr_); }
    TensorVarPtr<T> ptr() const { return ptr_; }

    // Indexing convenience, forwards to TensorVar
    T& operator[](sizeT idx) { return (*ptr_)[idx]; }
    const T& operator[](sizeT idx) const { return (*ptr_)[idx]; }

  private:
    TensorVarPtr<T> ptr_{};
};

template <typename T> class TensorVar : public std::enable_shared_from_this<TensorVar<T>>
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
    T& operator[](sizeT idx) { return getTensorData_()[idx]; }
    const T& operator[](sizeT idx) const { return getTensorData_()[idx]; }

    // Access underlying Tensor reference
    ml::TensorData<T>& tensorData() { return getTensorData_(); }
    const ml::TensorData<T>& tensorData() const { return getTensorData_(); }

    // Data view (const, matches Tensor<T>::data API)
    [[nodiscard]] const Vector<T>& data() const { return getTensorData_().data(); }

    // Frequently used operations (graph-building)
    // Helpers
    Tensor<T> unaryOp_(const TensorVarOpType op)
    {
        auto self = this->shared_from_this();
        return Tensor<T>(std::make_shared<TensorVar<T>>(op, std::initializer_list<TensorVarPtr<T>>{self}));
    }
    Tensor<T> binaryOp_(const TensorVarOpType op, const Tensor<T>& rhs)
    {
        auto self = this->shared_from_this();
        return Tensor<T>(std::make_shared<TensorVar<T>>(op, std::initializer_list<TensorVarPtr<T>>{self, rhs}));
    }

    // Reductions
    Tensor<T> sum() { return unaryOp_(TensorVarOpType::Sum); }
    Tensor<T> mean() { return unaryOp_(TensorVarOpType::Mean); }
    Tensor<T> max() { return unaryOp_(TensorVarOpType::Max); }
    Tensor<T> min() { return unaryOp_(TensorVarOpType::Min); }
    Tensor<T> prod() { return unaryOp_(TensorVarOpType::Prod); }

    // Unary elementwise
    Tensor<T> neg() { return unaryOp_(TensorVarOpType::Neg); }
    Tensor<T> absOp() { return unaryOp_(TensorVarOpType::Abs); }
    Tensor<T> sqrtOp() { return unaryOp_(TensorVarOpType::Sqrt); }
    Tensor<T> expOp() { return unaryOp_(TensorVarOpType::Exp); }
    Tensor<T> logOp() { return unaryOp_(TensorVarOpType::Log); }
    Tensor<T> sigmoid() { return unaryOp_(TensorVarOpType::Sigmoid); }

    // Binary elementwise
    Tensor<T> add(const Tensor<T>& rhs) { return binaryOp_(TensorVarOpType::Add, rhs); }
    Tensor<T> sub(const Tensor<T>& rhs) { return binaryOp_(TensorVarOpType::Sub, rhs); }
    Tensor<T> mul(const Tensor<T>& rhs) { return binaryOp_(TensorVarOpType::Mul, rhs); }
    Tensor<T> div(const Tensor<T>& rhs) { return binaryOp_(TensorVarOpType::Div, rhs); }
    Tensor<T> matmul(const Tensor<T>& rhs) { return binaryOp_(TensorVarOpType::MatMul, rhs); }
    // Immediate evaluation helpers (no graph)
    // sumValue(): immediate value from underlying TensorData
    T sumValue() const { return getTensorData_().sum(); }
    // transpose(): immediate TensorData result (graph op not defined yet)
    ml::TensorData<T> transpose() const { return getTensorData_().transpose(); }
    [[nodiscard]] sizeT dim() const { return getTensorData_().dim(); }
    [[nodiscard]] bool isScalar() const { return getTensorData_().isScalar(); }
    [[nodiscard]] bool hasOnlyOneVal() const { return getTensorData_().hasOnlyOneVal(); }
    [[nodiscard]] bool empty() const { return getTensorData_().empty(); }
    [[nodiscard]] auto rawData() const { return getTensorData_().rawData(); }

    // Indexing helpers
    [[nodiscard]] sizeT index(const std::initializer_list<sizeT> idxs) const { return getTensorData_().index(idxs); }
    [[nodiscard]] sizeT index(const Vector<sizeT>& idxs) const { return getTensorData_().index(idxs); }

    // Element access via coordinates
    T& operator()(const Vector<sizeT>& idxs) { return getTensorData_()(idxs); }
    const T& operator()(const Vector<sizeT>& idxs) const { return getTensorData_()(idxs); }

    template <typename... Dims>
    T& operator()(Dims... dims) { return getTensorData_()(std::forward<Dims>(dims)...); }
    template <typename... Dims>
    const T& operator()(Dims... dims) const { return getTensorData_()(std::forward<Dims>(dims)...); }

    // Setter-style call: v->operator()({i, j, ...}, value)
    void operator()(const std::initializer_list<sizeT> idxs, const T& value)
    {
        getTensorData_().set(idxs, value);
    }

    // Slicing and mutation
    ml::TensorData<T> at(const std::initializer_list<sizeT> idxs) const { return getTensorData_().at(idxs); }
    void set(const std::initializer_list<sizeT> idxs, T value) { getTensorData_().set(idxs, value); }
    T& first() { return getTensorData_().first(); }
    void fill(const T v) { getTensorData_().fill(v); }
    void copy(const ml::TensorData<T>& other) { getTensorData_().copy(other); }
    void copy(const Vector<T>& other) { getTensorData_().copy(other); }

    // Linalg and reshape
    T dot(const ml::TensorData<T>& other) const { return getTensorData_().dot(other); }
    T dot(const TensorVar& other) const { return getTensorData_().dot(other.tensorData()); }
    ml::TensorData<T> matmul(const ml::TensorData<T>& other) const { return getTensorData_().matmul(other); }
    ml::TensorData<T> matmul(const TensorVar& other) const { return getTensorData_().matmul(other.tensorData()); }
    void reshape(const Vector<sizeT>& new_shape) { getTensorData_().reshape(new_shape); }
    void print() const { getTensorData_().print(); }

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
    return Tensor<T>(std::make_shared<TensorVar<T>>(std::forward<Args>(args)...));
}

// Friendly overloads to support brace-init without verbose types
// 1) shape-only
template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<U>>>
Tensor<T> tensor(std::initializer_list<U> shape)
{
    Vector<sizeT> s;
    s.reserve(shape.size());
    for (auto v : shape) s.pushBack(static_cast<sizeT>(v));
    return Tensor<T>(std::make_shared<TensorVar<T>>(s));
}

// 2) shape + name
template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<U>>>
Tensor<T> tensor(std::initializer_list<U> shape, const char* name)
{
    Vector<sizeT> s;
    s.reserve(shape.size());
    for (auto v : shape) s.pushBack(static_cast<sizeT>(v));
    return Tensor<T>(std::make_shared<TensorVar<T>>(s, name));
}

// 3) shape + data
template <typename T, typename U>
Tensor<T> tensor(std::initializer_list<U> shape,
                 std::initializer_list<T> data)
{
    Vector<sizeT> s;
    s.reserve(shape.size());
    for (auto v : shape) s.pushBack(static_cast<sizeT>(v));
    return Tensor<T>(std::make_shared<TensorVar<T>>(s, data.begin()));
}

// 4) shape + data + name
template <typename T, typename U>
Tensor<T> tensor(std::initializer_list<U> shape,
                 std::initializer_list<T> data,
                 const char* name)
{
    Vector<sizeT> s;
    s.reserve(shape.size());
    for (auto v : shape) s.pushBack(static_cast<sizeT>(v));
    return Tensor<T>(std::make_shared<TensorVar<T>>(s, data.begin(), name));
}

// 5) scalar
template <typename T>
Tensor<T> tensor(T scalar)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(scalar));
}

// 6) scalar + name
template <typename T>
Tensor<T> tensor(T scalar, const char* name)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(scalar, name));
}

template<typename T>
Tensor<T> operator+ (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(TensorVarOpType::Add, {lhs, rhs}));
}

template<typename T>
Tensor<T> operator- (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(TensorVarOpType::Sub, {lhs, rhs}));
}

template<typename T>
Tensor<T> operator* (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(TensorVarOpType::Mul, {lhs, rhs}));
}

template<typename T>
Tensor<T> operator/ (const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return Tensor<T>(std::make_shared<TensorVar<T>>(TensorVarOpType::Div, {lhs, rhs}));
}


// -----------------------------------------------------------------------------
// Convenience free functions for shared_ptr<TensorVar<T>> ergonomics
// -----------------------------------------------------------------------------
template <typename T>
inline void set(const Tensor<T>& t,
                const std::initializer_list<sizeT> idxs,
                const T& value)
{
    t->set(idxs, value);
}

template <typename T>
inline T& atRef(const Tensor<T>& t, const std::initializer_list<sizeT> idxs)
{
    // Returns a reference via operator() to allow write access
    return const_cast<ml::TensorData<T>&>(static_cast<const ml::TensorData<T>&>(*t))(Vector<sizeT>(idxs));
}



} // namespace hahaha

#endif // HAHAHA_TENSORVAR_H
