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

#ifndef HAHAHA_MATH_TENSOR_WRAPPER_H
#define HAHAHA_MATH_TENSOR_WRAPPER_H

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common/Config.h"
#include "math/ds/TensorData.h"

class TensorWrapperTest;

namespace hahaha::math {

/**
 * @brief Main Tensor class providing a high-level API for numerical operations.
 *
 * This class wraps TensorData and provides a comprehensive set of operations
 * including basic arithmetic, matrix multiplication, and reshape capabilities.
 * It is designed to work with automatic differentiation and computational
 * graphs.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class TensorWrapper {
  public:
    /**
     * @brief Default constructor for an empty tensor.
     */
    TensorWrapper() = default;

    /**
     * @brief Copy constructor is deleted (use explicit copy if needed).
     */
    TensorWrapper(const TensorWrapper&) = delete;

    /**
     * @brief Move constructor.
     * @param other The source tensor to move from.
     */
    TensorWrapper(TensorWrapper&& other) noexcept
        : data_(std::move(other.data_)){
    }

    /**
     * @brief Copy assignment is deleted.
     */
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source tensor to move from.
     * @return Tensor& reference to this.
     */
    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            requiresGrad_ = other.requiresGrad_;
        }
        return *this;
    }

    /**
     * @brief Destructor.
     */
    ~TensorWrapper() = default;

    /**
     * @brief Construct from NestedData (e.g. nested initializer list).
     * @param data The source nested data.
     */
    // NOLINTNEXTLINE google-explicit-constructor
    TensorWrapper(NestedData<T>&& data) : data_(std::move(data)) {
    }

    /**
     * @brief Get a reference to the raw data pointer.
     * @return std::unique_ptr<T[]>& reference to internal data.
     */
    std::unique_ptr<T[]>& getRawData() {
        return data_.data_;
    }

    /**
     * @brief Get the tensor's shape.
     * @return const TensorShape& reference to internal shape.
     */
    [[nodiscard]] const TensorShape& shape() const {
        return data_.shape_;
    }

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const TensorStride& stride() const {
        return data_.stride_;
    }

    /**
     * @brief Element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return T& reference to the element.
     */
    T& at(const std::initializer_list<size_t>& indices) {
        const auto& shapeDims = data_.shape_.dims();
        if (indices.size() != shapeDims.size()) {
            throw std::out_of_range("Dimension mismatch: expected "
                                    + std::to_string(shapeDims.size())
                                    + " indices, got "
                                    + std::to_string(indices.size()));
        }

        size_t linearIdx = 0;
        const auto* idxIt = indices.begin();
        const auto& strideDims = data_.stride_.dims();

        for (size_t i = 0; i < shapeDims.size(); ++i) {
            size_t dimIdx = *idxIt;
            if (dimIdx >= shapeDims[i]) {
                throw std::out_of_range("Index out of bounds at dimension "
                                        + std::to_string(i));
            }
            linearIdx += dimIdx * strideDims[i];
            std::advance(idxIt, 1);
        }
        return data_.data_[linearIdx];
    }

    /**
     * @brief Constant element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return const T& reference to the element.
     */
    const T& at(const std::initializer_list<size_t>& indices) const {
        const auto& shapeDims = data_.shape_.dims();
        if (indices.size() != shapeDims.size()) {
            throw std::out_of_range("Dimension mismatch");
        }

        size_t linearIdx = 0;
        const auto* idxIt = indices.begin();
        const auto& strideDims = data_.stride_.dims();

        for (size_t i = 0; i < shapeDims.size(); ++i) {
            size_t dimIdx = *idxIt;
            if (dimIdx >= shapeDims[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            linearIdx += dimIdx * strideDims[i];
            std::advance(idxIt, 1);
        }
        return data_.data_[linearIdx];
    }

    /**
     * @brief Reshape tensor to new dimensions.
     * @param newShape Vector of new dimension sizes.
     * @return Tensor<T> A new tensor with reshaped dimensions.
     */
    TensorWrapper<T> reshape(const std::vector<size_t>& newShape) const {
        size_t totalSize = std::accumulate(
            newShape.begin(), newShape.end(), 1ULL, std::multiplies<size_t>());
        if (totalSize != size()) {
            throw std::invalid_argument(
                "New shape total size (" + std::to_string(totalSize)
                + ") must match current size (" + std::to_string(size()) + ")");
        }

        TensorWrapper<T> result;
        result.data_.shape_ = TensorShape(newShape);
        result.data_.stride_ = TensorStride(result.data_.shape_);

        size_t currentSize = size();
        result.data_.data_ = std::make_unique<T[]>(currentSize);
        std::copy(data_.data_.get(),
                  data_.data_.get() + currentSize,
                  result.data_.data_.get());

        return result;
    }

    /**
     * @brief Total number of elements in the tensor.
     * @return size_t count of elements.
     */
    [[nodiscard]] size_t size() const {
        return static_cast<size_t>(data_.shape_.totalSize());
    }

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t dimensions() const {
        return data_.shape_.dims().size();
    }

    /**
     * @brief Element-wise addition.
     * @param other The tensor to add.
     * @return Tensor<T> result tensor.
     */
    TensorWrapper<T> add(const TensorWrapper<T>& other) const {
        if (shape() != other.shape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for addition");
        }

        TensorWrapper<T> result;
        result.data_.shape_ = data_.shape_;
        result.data_.stride_ = data_.stride_;

        size_t tensorSize = size();
        result.data_.data_ = std::make_unique<T[]>(tensorSize);

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.data_[i] = data_.data_[i] + other.data_.data_[i];
        }

        return result;
    }

    /**
     * @brief Element-wise subtraction.
     * @param other The tensor to subtract.
     * @return Tensor<T> result tensor.
     */
    TensorWrapper<T> subtract(const TensorWrapper<T>& other) const {
        if (shape() != other.shape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for subtraction");
        }

        TensorWrapper<T> result;
        result.data_.shape_ = data_.shape_;
        result.data_.stride_ = data_.stride_;

        size_t tensorSize = size();
        result.data_.data_ = std::make_unique<T[]>(tensorSize);

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.data_[i] = data_.data_[i] - other.data_.data_[i];
        }

        return result;
    }

    /**
     * @brief Element-wise multiplication.
     * @param other The tensor to multiply.
     * @return Tensor<T> result tensor.
     */
    TensorWrapper<T> multiply(const TensorWrapper<T>& other) const {
        if (shape() != other.shape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for multiplication");
        }

        TensorWrapper<T> result;
        result.data_.shape_ = data_.shape_;
        result.data_.stride_ = data_.stride_;

        size_t tensorSize = size();
        result.data_.data_ = std::make_unique<T[]>(tensorSize);

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.data_[i] = data_.data_[i] * other.data_.data_[i];
        }

        return result;
    }

    /**
     * @brief Element-wise division.
     * @param other The tensor to divide.
     * @return Tensor<T> result tensor.
     */
    TensorWrapper<T> divide(const TensorWrapper<T>& other) const {
        if (shape() != other.shape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for division");
        }

        TensorWrapper<T> result;
        result.data_.shape_ = data_.shape_;
        result.data_.stride_ = data_.stride_;

        size_t tensorSize = size();
        result.data_.data_ = std::make_unique<T[]>(tensorSize);

        for (size_t i = 0; i < tensorSize; ++i) {
            if (other.data_.data_[i] == T(0)) {
                throw std::runtime_error("Division by zero at index "
                                         + std::to_string(i));
            }
            result.data_.data_[i] = data_.data_[i] / other.data_.data_[i];
        }

        return result;
    }

    /**
     * @brief Matrix multiplication (for 2D tensors).
     * @param other The tensor to multiply with.
     * @return Tensor<T> result tensor.
     */
    TensorWrapper<T> matmul(const TensorWrapper<T>& other) const {
        if (dimensions() != 2 || other.dimensions() != 2) {
            throw std::invalid_argument(
                "matmul is only implemented for 2D tensors");
        }

        const auto& thisDims = data_.shape_.dims();
        const auto& otherDims = other.data_.shape_.dims();

        if (thisDims[1] != otherDims[0]) {
            throw std::invalid_argument(
                "Matrix dimensions mismatch for matmul: ("
                + std::to_string(thisDims[0]) + "x"
                + std::to_string(thisDims[1]) + ") and ("
                + std::to_string(otherDims[0]) + "x"
                + std::to_string(otherDims[1]) + ")");
        }

        size_t rows = thisDims[0];
        size_t cols = otherDims[1];
        size_t inner = thisDims[1];

        TensorWrapper<T> result;
        result.data_.shape_ = TensorShape({rows, cols});
        result.data_.stride_ = TensorStride(result.data_.shape_);
        result.data_.data_ = std::make_unique<T[]>(rows * cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < inner; ++k) {
                    sum += data_.data_[i * inner + k]
                        * other.data_.data_[k * cols + j];
                }
                result.data_.data_[i * cols + j] = sum;
            }
        }

        return result;
    }

    /**
     * @brief Transpose operation.
     * @return Tensor<T> transposed tensor.
     */
    TensorWrapper<T> transpose() const {
        if (dimensions() != 2) {
            throw std::invalid_argument(
                "transpose is only implemented for 2D tensors for now");
        }

        const auto& shapeDims = data_.shape_.dims();
        size_t rows = shapeDims[0];
        size_t cols = shapeDims[1];

        TensorWrapper<T> result;
        result.data_.shape_ = TensorShape({cols, rows});
        result.data_.stride_ = TensorStride(result.data_.shape_);
        result.data_.data_ = std::make_unique<T[]>(size());

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_.data_[j * rows + i] = data_.data_[i * cols + j];
            }
        }

        return result;
    }

    /**
     * @brief Broadcast tensor to match the shape of another tensor.
     * @param other The target tensor for broadcasting.
     */
    void broadcast(const TensorWrapper<T>& /*other*/) {
        throw std::runtime_error("Broadcasting not implemented yet");
    }

    TensorWrapper<T> operator+(const TensorWrapper<T>& other) const {
        return add(other);
    }
    TensorWrapper<T> operator-(const TensorWrapper<T>& other) const {
        return subtract(other);
    }
    TensorWrapper<T> operator*(const TensorWrapper<T>& other) const {
        return multiply(other);
    }
    TensorWrapper<T> operator/(const TensorWrapper<T>& other) const {
        return divide(other);
    }
    TensorWrapper<T> operator-() const {
        TensorWrapper<T> result;
        result.data_.shape_ = data_.shape_;
        result.data_.stride_ = data_.stride_;
        size_t tensorSize = size();
        result.data_.data_ = std::make_unique<T[]>(tensorSize);
        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.data_[i] = -data_.data_[i];
        }
        return result;
    }

  private:
    TensorData<T> data_;

    bool requiresGrad_ = common::getConfig().defaultRequiresGrad;
    std::shared_ptr<TensorWrapper<T>> grad_ = nullptr;

    // default name of the test fixture class to this class
    friend class ::TensorWrapperTest;
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_H
