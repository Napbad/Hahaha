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
// Created by Napbad on 7/27/25.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "Error.h"
#include "Res.h"
#include "defines/h3defs.h"
#include "ds/Vector.h"

namespace hahaha::ml
{

using namespace hahaha::core;

class TensorErr final : public BaseError
{
  public:
    TensorErr() = default;
    explicit TensorErr(const char* msg) : BaseError(msg)
    {
    }
    explicit TensorErr(const ds::String& msg) : BaseError(msg)
    {
    }
};

template <typename T> class Tensor
{
  public:
    using ValueType = T;

    Tensor() = default;

    Tensor(const std::initializer_list<sizeT> shape)
        : shape_(shape), data_(computeSize(shape))
    {
    }

    explicit Tensor(const ds::Vector<sizeT>& shape)
        : shape_(shape), data_(computeSize(shape))
    {
    }

    // Constructor for a 0-dimensional tensor (scalar)
    explicit Tensor(const std::initializer_list<sizeT> shape,
                    std::initializer_list<T> data)
        : shape_(shape), data_(data)
    {
    }

    // Access shape
    [[nodiscard]] const ds::Vector<sizeT>& shape() const
    {
        return shape_;
    }
    [[nodiscard]] sizeT size() const
    {
        return data_.size();
    }

    // Index calculation (flattened)
    [[nodiscard]] Res<sizeT, BaseError>
    index(const std::initializer_list<sizeT> indices) const
    {
        SetRetT(sizeT, BaseError)

            if (indices.size() != shape_.size())
                Err(BaseError("Incorrect number of indices"));
        sizeT idx = 0;
        sizeT stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i)
        {
            if (indices.begin()[i] >= shape_[i])
            {
                Err(BaseError("Index out of bounds"));
            }
            idx += indices.begin()[i] * stride;
            stride *= shape_[i];
        }
        Ok(idx);
    }

    [[nodiscard]] Res<sizeT, BaseError>
    index(const ds::Vector<sizeT>& indices) const
    {
        SetRetT(sizeT, BaseError)

            if (indices.size() != shape_.size())
        {
            Err(BaseError("Incorrect number of indices"));
        }
        sizeT idx = 0;
        sizeT stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= shape_[i])
            {
                Err(BaseError("Index out of bounds"));
            }
            idx += indices[i] * stride;
            stride *= shape_[i];
        }
        Ok(idx);
    }

    // Element access
    ValueType& operator()(const ds::Vector<sizeT>& indices)
    {
        return data_[index(indices).unwrap()];
    }
    const ValueType& operator()(const ds::Vector<sizeT>& indices) const
    {
        return data_[index(indices).unwrap()];
    }

    // Fill tensor with value
    void fill(const ValueType v)
    {
        std::ranges::fill(data_, v);
    }

    // Element-wise addition
    Tensor operator+(const Tensor& other) const
    {
        checkShapeAndSizeNotEqual(other);
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    // Element-wise multiplication
    Tensor operator*(const Tensor& other) const
    {
        checkShapeAndSizeNotEqual(other);
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    // Scalar operations
    Tensor operator+(ValueType scalar) const
    {
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] + scalar;
        }
        return result;
    }

    Tensor operator*(ValueType scalar) const
    {
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    T dot(const Tensor& other) const
    {
        checkShapeAndSizeNotEqual(other);
        T result = 0;
        for (sizeT i = 0; i < size(); ++i)
        {
            result += data_[i] * other.data_[i];
        }
        return result;
    }

    // Print
    void printFlat() const
    {
        for (const auto& v : data_)
        {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }

    T& first()
    {
        return data_[0];
    }

    // Static factory methods
    static Tensor fromVector(const ds::Vector<T>& vec)
    {
        Tensor tensor({vec.size()});
        for (sizeT i = 0; i < vec.size(); ++i)
        {
            tensor.data_[i] = vec[i];
        }
        return tensor;
    }

    Res<void, TensorErr> copy(const Tensor& other)
    {
        SetRetT(void, TensorErr) if (other.shape() != shape_)
        {
            Err(TensorErr(
                "Cannot copy value from a tensor with different shape"))
        }
        for (sizeT i = 0; i < other.size(); ++i)
        {
            data_[i] = other.data_[i];
        }
        Ok()
    }

    Res<void, TensorErr> copy(const ds::Vector<T>& other)
    {
        SetRetT(void, TensorErr) if (other.size() != this->size())
        {
            Err(TensorErr(
                "Cannot copy value from a tensor with different shape"))
        }
        for (sizeT i = 0; i < other.size(); ++i)
        {
            data_[i] = other[i];
        }
        Ok()
    }

    [[nodiscard]] sizeT dim() const
    {
        return shape().size();
    }

    // Element-wise subtraction
    Tensor operator-(const Tensor& other) const
    {
        checkShapeAndSizeNotEqual(other);
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    // Element-wise division
    Tensor operator/(const Tensor& other) const
    {
        checkShapeAndSizeNotEqual(other);
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            // Add check for division by zero if T is a floating-point type
            if constexpr (std::is_floating_point_v<T>)
            {
                if (other.data_[i] == static_cast<T>(0))
                {
                    throw std::runtime_error("Division by zero");
                }
            }
            result.data_[i] = data_[i] / other.data_[i];
        }
        return result;
    }

    // Scalar operations (subtraction and division)
    Tensor operator-(ValueType scalar) const
    {
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] - scalar;
        }
        return result;
    }

    Tensor operator/(ValueType scalar) const
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            if (scalar == static_cast<T>(0))
            {
                throw std::runtime_error("Division by zero by scalar");
            }
        }
        Tensor result(shape_);
        for (sizeT i = 0; i < size(); ++i)
        {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Compound assignment operators
    Tensor& operator+=(const Tensor& other)
    {
        checkShapeAndSizeNotEqual(other);
        for (sizeT i = 0; i < size(); ++i)
        {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other)
    {
        checkShapeAndSizeNotEqual(other);
        for (sizeT i = 0; i < size(); ++i)
        {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Tensor& operator*=(const Tensor& other)
    {
        checkShapeAndSizeNotEqual(other);
        for (sizeT i = 0; i < size(); ++i)
        {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    Tensor& operator/=(const Tensor& other)
    {
        checkShapeAndSizeNotEqual(other);
        for (sizeT i = 0; i < size(); ++i)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                if (other.data_[i] == static_cast<T>(0))
                {
                    throw std::runtime_error("Division by zero");
                }
            }
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    [[nodiscard]] auto begin() const
    {
        return data_.begin();
    }
    [[nodiscard]] auto end() const
    {
        return data_.end();
    }

    [[nodiscard]] auto rawData() const
    {
        return data_;
    }

    [[nodiscard]] bool empty() const
    {
        return data_.empty();
    }

    // Array-style element access (read-only)
    const T& operator[](sizeT index) const
    {
        return data_[index];
    }

    // Array-style element access (read-write)
    T& operator[](sizeT index)
    {
        return data_[index];
    }

    // Sum all elements in the tensor
    T sum() const
    {
        T result = static_cast<T>(0);
        for (sizeT i = 0; i < size(); ++i)
        {
            result += data_[i];
        }
        return result;
    }

    Res<Tensor, IndexOutOfBoundError>
    at(const std::initializer_list<sizeT> indices) const
    {
        SetRetT(Tensor, IndexOutOfBoundError) if (indices.size() > dim())
        {
            Err("Too many indices for at() method");
        }
        if (indices.size() == 0 && dim() != 0)
        {
            Err("Cannot access scalar tensor with empty indices")
        }
        // calculate start index and tensor size
        auto idxRes = index(indices);
        if (idxRes.isErr())
        {
            Err(IndexOutOfBoundError(idxRes.unwrapErr().message()));
        }

        sizeT start = idxRes.unwrap();
        sizeT length = 1;

        if (indices.size() == dim())
        {
            Ok(Tensor({0}, {data_[start]}))
        }

        ds::Vector<sizeT> newShape;
        for (sizeT i = indices.size(); i < shape_.size(); ++i)
        {
            newShape.pushBack(shape_[i]);
            length *= shape_[i];
        }

        if (start + length > data_.size())
        {
            Err(IndexOutOfBoundError("Calculated range out of bounds"));
        }

        Tensor res(newShape);
        // res.replaceSelf(data_.begin() + start, data_.begin() + start +
        // length);
        res.copy(data_.subVector(start, length));
        Ok(res);
    }

    Res<void, IndexOutOfBoundError>
    set(const std::initializer_list<sizeT> indices, T value)
    {
        SetRetT(void, IndexOutOfBoundError) if (indices.size() > dim())
        {
            Err("Too many indices for set() method");
        }
        if (indices.size() == 0 && dim() != 0)
        {
            Err("Cannot access scalar tensor with empty indices")
        }
        // calculate start index and tensor size
        auto idxRes = index(indices);
        if (idxRes.isErr())
        {
            Err(IndexOutOfBoundError(idxRes.unwrapErr().message()));
        }

        data_[idxRes.unwrap()] = value;
        Ok()
    }

  private:
    ds::Vector<sizeT> shape_;
    ds::Vector<ValueType> data_;

    static sizeT computeSize(const ds::Vector<sizeT>& shape)
    {
        return std::accumulate(
            shape.begin(), shape.end(), sizeT{1}, std::multiplies<>());
    }

    void checkShapeAndSizeNotEqual(const Tensor& other) const
    {
        if (shape_ != other.shape_)
        {
            throw std::runtime_error("Shape mismatch");
        }
    }
};

} // namespace hahaha::ml
#endif // TENSOR_H
