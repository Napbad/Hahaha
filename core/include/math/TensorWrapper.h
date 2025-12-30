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
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "math/ds/TensorData.h"
#include "math/ds/TensorShape.h"

class TensorWrapperTest;

namespace hahaha::compute {
template <typename T> class ComputeNode;
}

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
     * @brief Construct a tensor with a given shape and initial value.
     * @param shape The shape of the tensor.
     * @param initValue The initial value for all elements.
     */
    explicit TensorWrapper(const TensorShape& shape, T initValue = 0)
        : data_(TensorData<T>(shape, initValue)) {}

    /**
     * @brief Copy constructor. Performs a deep copy of the data.
     * @param other The tensor to copy from.
     */
    TensorWrapper(const TensorWrapper& other) : data_(other.data_) {}

    /**
     * @brief Move constructor. Transfers ownership of the data.
     * @param other The source tensor to move from.
     */
    TensorWrapper(TensorWrapper&& other) noexcept
        : data_(std::move(other.data_)) {}

    /**
     * @brief Copy assignment is deleted to encourage explicit copying.
     */
    TensorWrapper& operator=(const TensorWrapper&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source tensor to move from.
     * @return TensorWrapper& reference to this.
     */
    TensorWrapper& operator=(TensorWrapper&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    /**
     * @brief Destructor.
     */
    ~TensorWrapper() = default;

    /**
     * @brief Construct from NestedData (e.g., nested initializer list).
     * @param data The source nested data.
     */
    explicit TensorWrapper(NestedData<T>&& data) : data_(std::move(data)) {}

    /**
     * @brief Get a reference to the raw data pointer.
     * @return Reference to the unique_ptr holding the data array.
     */
    std::unique_ptr<T[]>& getRawData() { return data_.getData(); }

    /**
     * @brief Get the tensor's shape.
     * @return const TensorShape& reference to internal shape.
     */
    [[nodiscard]] const TensorShape& getShape() const { return data_.getShape(); }

    /**
     * @brief Get the tensor's strides.
     * @return const TensorStride& reference to internal strides.
     */
    [[nodiscard]] const TensorStride& getStride() const {
        return data_.getStride();
    }

    /**
     * @brief Element access with bounds checking.
     *
     * Formula for linear index in row-major:
     * index = sum(indices[i] * strides[i])
     *
     * @param indices List of indices for each dimension.
     * @return T& reference to the element.
     */
    T& at(const std::initializer_list<size_t>& indices) {
        const auto& shapeDims = data_.getShape().getDims();
        if (indices.size() != shapeDims.size()) {
            throw std::out_of_range("Dimension mismatch: expected " +
                                    std::to_string(shapeDims.size()) +
                                    " indices, got " +
                                    std::to_string(indices.size()));
        }

        size_t linearIdx = 0;
        const auto* idxIt = indices.begin();
        const auto& strideDims = data_.getStride().getDims();

        for (size_t i = 0; i < shapeDims.size(); ++i) {
            size_t dimIdx = *idxIt;
            if (dimIdx >= shapeDims[i]) {
                throw std::out_of_range("Index out of bounds at dimension " +
                                        std::to_string(i));
            }
            linearIdx += dimIdx * strideDims[i];
            std::advance(idxIt, 1);
        }
        return data_.getData()[linearIdx];
    }

    /**
     * @brief Constant element access with bounds checking.
     * @param indices List of indices for each dimension.
     * @return const T& reference to the element.
     */
    const T& at(const std::initializer_list<size_t>& indices) const {
        const auto& shapeDims = data_.getShape().getDims();
        if (indices.size() != shapeDims.size()) {
            throw std::out_of_range("Dimension mismatch");
        }

        size_t linearIdx = 0;
        const auto* idxIt = indices.begin();
        const auto& strideDims = data_.getStride().getDims();

        for (size_t i = 0; i < shapeDims.size(); ++i) {
            size_t dimIdx = *idxIt;
            if (dimIdx >= shapeDims[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            linearIdx += dimIdx * strideDims[i];
            std::advance(idxIt, 1);
        }
        return data_.getData()[linearIdx];
    }

    /**
     * @brief Reshape tensor to new dimensions.
     *
     * Total size must remain invariant.
     *
     * @param newShape Vector of new dimension sizes.
     * @return TensorWrapper<T> A new tensor with reshaped dimensions.
     */
    TensorWrapper<T> reshape(const std::vector<size_t>& newShape) const {
        size_t totalSize = std::accumulate(newShape.begin(), newShape.end(),
                                           1ULL, std::multiplies<size_t>());
        if (totalSize != getSize()) {
            throw std::invalid_argument(
                "New shape total size (" + std::to_string(totalSize) +
                ") must match current size (" + std::to_string(getSize()) + ")");
        }

        TensorWrapper<T> result;
        result.data_.setShape(TensorShape(newShape));
        result.data_.setStride(TensorStride(result.data_.getShape()));

        size_t currentSize = getSize();
        result.data_.setData(std::make_unique<T[]>(currentSize));
        std::copy(data_.getData().get(), data_.getData().get() + currentSize,
                  result.data_.getData().get());

        return result;
    }

    /**
     * @brief Total number of elements in the tensor.
     * @return size_t count of elements.
     */
    [[nodiscard]] size_t getSize() const {
        return static_cast<size_t>(data_.getShape().getTotalSize());
    }

    /**
     * @brief Number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t getDimensions() const {
        return data_.getShape().getDims().size();
    }

    /**
     * @brief Element-wise addition.
     *
     * Formula: res[i] = a[i] + b[i]
     *
     * @param other The tensor to add.
     * @return TensorWrapper<T> result tensor.
     */
    TensorWrapper<T> add(const TensorWrapper<T>& other) const {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for addition");
        }

        TensorWrapper<T> result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());

        size_t tensorSize = getSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.getData()[i] =
                data_.getData()[i] + other.data_.getData()[i];
        }

        return result;
    }

    /**
     * @brief Element-wise subtraction.
     *
     * Formula: res[i] = a[i] - b[i]
     *
     * @param other The tensor to subtract.
     * @return TensorWrapper<T> result tensor.
     */
    TensorWrapper<T> subtract(const TensorWrapper<T>& other) const {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for subtraction");
        }

        TensorWrapper<T> result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());

        size_t tensorSize = getSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.getData()[i] =
                data_.getData()[i] - other.data_.getData()[i];
        }

        return result;
    }

    /**
     * @brief Element-wise multiplication.
     *
     * Formula: res[i] = a[i] * b[i]
     *
     * @param other The tensor to multiply.
     * @return TensorWrapper<T> result tensor.
     */
    TensorWrapper<T> multiply(const TensorWrapper<T>& other) const {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for multiplication");
        }

        TensorWrapper<T> result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());

        size_t tensorSize = getSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));

        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.getData()[i] =
                data_.getData()[i] * other.data_.getData()[i];
        }

        return result;
    }

    /**
     * @brief Element-wise division.
     *
     * Formula: res[i] = a[i] / b[i]
     *
     * @param other The tensor to divide.
     * @return TensorWrapper<T> result tensor.
     */
    TensorWrapper<T> divide(const TensorWrapper<T>& other) const {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for division");
        }

        TensorWrapper<T> result;
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());

        size_t tensorSize = getSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));

        for (size_t i = 0; i < tensorSize; ++i) {
            if (other.data_.getData()[i] == T(0)) {
                throw std::runtime_error("Division by zero at index " +
                                         std::to_string(i));
            }
            result.data_.getData()[i] =
                data_.getData()[i] / other.data_.getData()[i];
        }

        return result;
    }

    /**
     * @brief Matrix multiplication (for 2D tensors).
     *
     * Formula: C[i, j] = sum(A[i, k] * B[k, j]) for k in 0..K-1
     * where A is (M x K) and B is (K x N).
     *
     * @param other The tensor to multiply with.
     * @return TensorWrapper<T> result tensor.
     */
    TensorWrapper<T> matmul(const TensorWrapper<T>& other) const {
        if (getDimensions() != 2 || other.getDimensions() != 2) {
            throw std::invalid_argument(
                "matmul is only implemented for 2D tensors");
        }

        const auto& thisDims = data_.getShape().getDims();
        const auto& otherDims = other.data_.getShape().getDims();

        if (thisDims[1] != otherDims[0]) {
            throw std::invalid_argument(
                "Matrix dimensions mismatch for matmul: (" +
                std::to_string(thisDims[0]) + "x" + std::to_string(thisDims[1]) +
                ") and (" + std::to_string(otherDims[0]) + "x" +
                std::to_string(otherDims[1]) + ")");
        }

        size_t rows = thisDims[0];
        size_t cols = otherDims[1];
        size_t inner = thisDims[1];

        TensorWrapper<T> result;
        result.data_.setShape(TensorShape({rows, cols}));
        result.data_.setStride(TensorStride(result.data_.getShape()));
        result.data_.setData(std::make_unique<T[]>(rows * cols));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < inner; ++k) {
                    sum += data_.getData()[i * inner + k] *
                           other.data_.getData()[k * cols + j];
                }
                result.data_.getData()[i * cols + j] = sum;
            }
        }

        return result;
    }

    /**
     * @brief Transpose operation (for 2D tensors).
     *
     * Formula: B[j, i] = A[i, j]
     *
     * @return TensorWrapper<T> transposed tensor.
     */
    TensorWrapper<T> transpose() const {
        if (getDimensions() != 2) {
            throw std::invalid_argument(
                "transpose is only implemented for 2D tensors for now");
        }

        const auto& shapeDims = data_.getShape().getDims();
        size_t rows = shapeDims[0];
        size_t cols = shapeDims[1];

        TensorWrapper<T> result;
        result.data_.setShape(TensorShape({cols, rows}));
        result.data_.setStride(TensorStride(result.data_.getShape()));
        result.data_.setData(std::make_unique<T[]>(getSize()));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_.getData()[j * rows + i] =
                    data_.getData()[i * cols + j];
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
        result.data_.setShape(data_.getShape());
        result.data_.setStride(data_.getStride());
        size_t tensorSize = getSize();
        result.data_.setData(std::make_unique<T[]>(tensorSize));
        for (size_t i = 0; i < tensorSize; ++i) {
            result.data_.getData()[i] = -data_.getData()[i];
        }
        return result;
    }

    TensorWrapper<T>& operator+=(const TensorWrapper<T>& other) {
        if (getShape() != other.getShape()) {
            throw std::invalid_argument(
                "Tensors must have the same shape for addition");
        }

        size_t tensorSize = getSize();

        for (size_t i = 0; i < tensorSize; ++i) {
            data_.getData()[i] = data_.getData()[i] + other.data_.getData()[i];
        }

        return *this;
    }

  private:
    TensorData<T> data_; /**< Managed tensor data and metadata. */

    // Friend classes for internal access
    friend class ::TensorWrapperTest;
    friend class hahaha::compute::ComputeNode<T>;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_WRAPPER_H
