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
#include <stdexcept>

#include "backend/Device.h"
#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h"
#include "math/ds/TensorStride.h"

namespace hahaha::math {

template <typename T> class TensorWrapper;

/**
 * @brief Internal storage class for tensor data and metadata.
 *
 * TensorData manages the raw memory allocation, the shape of the tensor,
 * and the memory strides. It uses std::unique_ptr for automatic memory
 * management on the CPU. For GPU, a separate memory management strategy
 * would be needed.
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
     * @brief Construct with given shape and initial value on a specific device.
     * @param shape The shape of the tensor.
     * @param initValue Initial value for all elements.
     * @param device The device where the data should reside.
     */
    TensorData(const TensorShape& shape,
               T initValue,
               backend::Device device = backend::Device())
        : shape_(shape), stride_(shape), device_(device) {
        size_t size = shape_.getTotalSize();
        if (device_.type == backend::DeviceType::CPU
            || device_.type == backend::DeviceType::SIMD) {
            data_ = std::make_unique<T[]>(size);
            std::fill(data_.get(), data_.get() + size, initValue);
        } else {
            // TODO: Handle GPU allocation using compute::gpu::GpuMemory
            throw std::runtime_error(
                "GPU allocation not yet implemented in TensorData");
        }
    }

    /**
     * @brief Copy constructor. Performs a deep copy of the underlying array.
     * @param other The TensorData to copy from.
     */
    TensorData(const TensorData& other)
        : shape_(other.shape_), stride_(other.stride_), device_(other.device_) {
        size_t size = shape_.getTotalSize();
        if (device_.type == backend::DeviceType::CPU
            || device_.type == backend::DeviceType::SIMD) {
            data_ = std::make_unique<T[]>(size);
            std::copy(other.data_.get(), other.data_.get() + size, data_.get());
        } else {
            // TODO: Handle GPU deep copy
            throw std::runtime_error(
                "GPU deep copy not yet implemented in TensorData");
        }
    }

    /**
     * @brief Move constructor. Efficiently transfers ownership of the array.
     * @param other The source TensorData to move from.
     */
    TensorData(TensorData&& other) noexcept
        : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
          stride_(std::move(other.stride_)), device_(other.device_) {
    }

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
            device_ = other.device_;
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
    explicit TensorData(NestedData<T>&& data)
        : shape_(data.getShape()), device_(backend::Device()) {
        size_t size = data.getFlatData().size();
        data_ = std::make_unique<T[]>(size);
        std::copy(
            data.getFlatData().begin(), data.getFlatData().end(), data_.get());
        stride_ = TensorStride(shape_);
    }

    /**
     * @brief Get the raw data pointer.
     * @return Reference to the unique_ptr holding the data.
     */
    std::unique_ptr<T[]>& getData() {
        return data_;
    }

    /**
     * @brief Const version of data pointer access.
     * @return Const reference to the unique_ptr.
     */
    const std::unique_ptr<T[]>& getData() const {
        return data_;
    }

    /**
     * @brief Replace the current data array.
     * @param data New data array as unique_ptr.
     */
    void setData(std::unique_ptr<T[]> data) {
        data_ = std::move(data);
    }

    /**
     * @brief Get the tensor shape.
     * @return Const reference to the shape.
     */
    [[nodiscard]] const TensorShape& getShape() const {
        return shape_;
    }

    /**
     * @brief Update the tensor shape.
     * @param shape New shape.
     */
    void setShape(const TensorShape& shape) {
        shape_ = shape;
    }

    /**
     * @brief Get the memory strides.
     * @return Const reference to the strides.
     */
    [[nodiscard]] const TensorStride& getStride() const {
        return stride_;
    }

    /**
     * @brief Update the memory strides.
     * @param stride New strides.
     */
    void setStride(const TensorStride& stride) {
        stride_ = stride;
    }

    /**
     * @brief Get the device where the data is stored.
     * @return Const reference to the device.
     */
    [[nodiscard]] const backend::Device& getDevice() const {
        return device_;
    }

    /**
     * @brief Set the device for this tensor data.
     * @param device New device.
     */
    void setDevice(const backend::Device& device) {
        device_ = device;
    }

  private:
    std::unique_ptr<T[]> data_; /**< Raw heap-allocated data array. */
    TensorShape shape_;         /**< Dimensionality metadata. */
    TensorStride stride_;       /**< Memory skip values for indexing. */
    backend::Device device_;    /**< Device where data resides. */

    friend class TensorWrapper<T>;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_DATA_H
