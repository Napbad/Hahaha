// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      https://www.apache.org/licenses/LICENSE-2.0 
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
// 

#ifndef HAHAHA_MATH_TENSOR_H
#define HAHAHA_MATH_TENSOR_H

#include <vector>
#include "math/ds/TensorData.h"

class TensorDataTest;

namespace hahaha::math
{

/**
 * @brief Main Tensor class providing a high-level API for numerical operations.
 *
 * This class wraps TensorData and provides a comprehensive set of operations
 * including basic arithmetic, matrix multiplication, and reshape capabilities.
 * It is designed to work with automatic differentiation and computational graphs.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class Tensor
{
  public:
    /**
     * @brief Default constructor for an empty tensor.
     */
    Tensor() = default;

    /**
     * @brief Copy constructor is deleted (use explicit copy if needed).
     */
    Tensor(const Tensor&) = delete;

    /**
     * @brief Move constructor.
     * @param other The source tensor to move from.
     */
    Tensor(Tensor&& other) noexcept 
        : data_(std::move(other.data_)), 
          forwardTensor_(std::move(other.forwardTensor_)) {}

    /**
     * @brief Copy assignment is deleted.
     */
    Tensor& operator=(const Tensor&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source tensor to move from.
     * @return Tensor& reference to this.
     */
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            forwardTensor_ = std::move(other.forwardTensor_);
        }
        return *this;
    }

    /**
     * @brief Destructor.
     */
    ~Tensor() = default;

    /**
     * @brief Construct from NestedData (e.g. nested initializer list).
     * @param data The source nested data.
     */
    // NOLINTNEXTLINE google-explicit-constructor
    Tensor(NestedData<T>&& data) : data_(std::move(data))
    {

    }

    /**
     * @brief Get a reference to the raw data pointer.
     * @return std::unique_ptr<T[]>& reference to internal data.
     */
    std::unique_ptr<T[]>& getRawData()
    {
        return data_.data_;
    }

    /**
     * @brief Get the tensor's shape.
     * @return const TensorShape& reference to internal shape.
     */
    [[nodiscard]] const TensorShape& shape() const
    {
        return data_.shape_;
    }

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const TensorStride& stride() const
    {
        return data_.stride_;
    }

    /**
     * @brief Element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return T& reference to the element.
     */
    T& at(const std::initializer_list<size_t>& indices);

    /**
     * @brief Constant element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return const T& reference to the element.
     */
    const T& at(const std::initializer_list<size_t>& indices) const;

    /**
     * @brief Reshape tensor to new dimensions.
     * @param newShape Vector of new dimension sizes.
     * @return Tensor<T> A new tensor with reshaped dimensions.
     */
    Tensor<T> reshape(const std::vector<size_t>& newShape) const;

    /**
     * @brief Total number of elements in the tensor.
     * @return size_t count of elements.
     */
    [[nodiscard]] size_t size() const;

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t dimensions() const;

    /**
     * @brief Element-wise addition.
     * @param other The tensor to add.
     * @return Tensor<T> result tensor.
     */
    Tensor<T> add(const Tensor<T>& other) const;

    /**
     * @brief Element-wise subtraction.
     * @param other The tensor to subtract.
     * @return Tensor<T> result tensor.
     */
    Tensor<T> subtract(const Tensor<T>& other) const;

    /**
     * @brief Element-wise multiplication.
     * @param other The tensor to multiply.
     * @return Tensor<T> result tensor.
     */
    Tensor<T> multiply(const Tensor<T>& other) const;

    /**
     * @brief Element-wise division.
     * @param other The tensor to divide.
     * @return Tensor<T> result tensor.
     */
    Tensor<T> divide(const Tensor<T>& other) const;

    /**
     * @brief Matrix multiplication (for 2D tensors).
     * @param other The tensor to multiply with.
     * @return Tensor<T> result tensor.
     */
    Tensor<T> matmul(const Tensor<T>& other) const;

    /**
     * @brief Transpose operation.
     * @return Tensor<T> transposed tensor.
     */
    Tensor<T> transpose() const;

    /**
     * @brief Broadcast tensor to match the shape of another tensor.
     * @param other The target tensor for broadcasting.
     */
    void broadcast(const Tensor<T>& other);

    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator/(const Tensor<T>& other) const;
    Tensor<T> operator-() const;

  private:
    TensorData<T> data_;

    // used for autograd and compute graph
    std::vector<TensorData<T>> forwardTensor_;

    // default name of the test fixture class to this class
    friend class ::TensorDataTest;

};
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_H