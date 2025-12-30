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

#ifndef HAHAHA_MATH_DS_TENSOR_DATA_H
#define HAHAHA_MATH_DS_TENSOR_DATA_H

#include <memory>

#include "common/definitions.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h"
#include "math/ds/TensorStride.h"
#include "utils/common/HelperStruct.h"

namespace hahaha::math {

template <typename T> class TensorWrapper;

/**
 * @brief Internal storage class for tensor data and metadata.
 *
 * TensorData manages the raw memory allocation, the shape of the tensor,
 * and the memory strides. It uses std::unique_ptr for automatic memory
 * management.
 *
 * This class is designed to be wrapped by TensorWrapper, which provides
 * the high-level API.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class TensorData {
  public:
    /**
     * @brief Default constructor for an empty storage.
     */
    TensorData() = default;

    /**
     * @brief Construct with given shape and initial value.
     * @param shape The shape of the tensor.
     * @param initValue Initial value for all elements.
     */
    TensorData(const TensorShape& shape, T initValue)
        : shape_(shape), stride_(shape) {
        size_t size = shape_.getTotalSize();
        data_ = std::make_unique<T[]>(size);
        std::fill(data_.get(), data_.get() + size, initValue);
    }

    /**
     * @brief Copy constructor. Performs a deep copy of the underlying array.
     * @param other The TensorData to copy from.
     */
    TensorData(const TensorData& other)
        : shape_(other.shape_), stride_(other.stride_) {
        size_t size = shape_.getTotalSize();
        data_ = std::make_unique<T[]>(size);
        std::copy(other.data_.get(), other.data_.get() + size, data_.get());
    }

    /**
     * @brief Move constructor. Efficiently transfers ownership of the array.
     * @param other The source TensorData to move from.
     */
    TensorData(TensorData&& other) noexcept
        : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
          stride_(std::move(other.stride_)) {}

    /**
     * @brief Copy assignment is deleted. Use constructor for copying.
     */
    TensorData& operator=(const TensorData&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source TensorData to move from.
     * @return Reference to this.
     */
    TensorData& operator=(TensorData&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = std::move(other.shape_);
            stride_ = std::move(other.stride_);
        }
        return *this;
    }

    /**
     * @brief Destructor. Automatically releases the unique_ptr.
     */
    ~TensorData() = default;

    /**
     * @brief Construct from flattened nested data (e.g., from initializer
     * lists).
     * @param data The NestedData object containing flattened data and shape.
     */
    explicit TensorData(NestedData<T>&& data) : shape_(data.getShape()) {
        size_t size = data.getFlatData().size();
        data_ = std::make_unique<T[]>(size);
        std::copy(data.getFlatData().begin(), data.getFlatData().end(),
                  data_.get());
        stride_ = TensorStride(shape_);
    }

    /**
     * @brief Get the raw data pointer.
     * @return Reference to the unique_ptr holding the data.
     */
    std::unique_ptr<T[]>& getData() { return data_; }

    /**
     * @brief Const version of data pointer access.
     * @return Const reference to the unique_ptr.
     */
    const std::unique_ptr<T[]>& getData() const { return data_; }

    /**
     * @brief Replace the current data array.
     * @param data New data array as unique_ptr.
     */
    void setData(std::unique_ptr<T[]> data) { data_ = std::move(data); }

    /**
     * @brief Get the tensor shape.
     * @return Const reference to the shape.
     */
    [[nodiscard]] const TensorShape& getShape() const { return shape_; }

    /**
     * @brief Update the tensor shape.
     * @param shape New shape.
     */
    void setShape(const TensorShape& shape) { shape_ = shape; }

    /**
     * @brief Get the memory strides.
     * @return Const reference to the strides.
     */
    [[nodiscard]] const TensorStride& getStride() const { return stride_; }

    /**
     * @brief Update the memory strides.
     * @param stride New strides.
     */
    void setStride(const TensorStride& stride) { stride_ = stride; }

  private:
    std::unique_ptr<T[]> data_; /**< Raw heap-allocated data array. */
    TensorShape shape_;         /**< Dimensionality metadata. */
    TensorStride stride_;       /**< Memory skip values for indexing. */

    friend class TensorWrapper<T>;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_DATA_H
