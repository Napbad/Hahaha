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

namespace hahaha::math
{

template <typename T> class Tensor;

using hahaha::common::u32;
using hahaha::utils::isInitList;
using hahaha::utils::isNestedInitList;

/**
 * @brief TensorData
 *
 * @tparam ValueType
 */
/**
 * @brief Manages the raw data, shape, and strides of a tensor.
 *
 * TensorData handles memory allocation (using unique_ptr) and provides
 * a bridge between nested input data and the core Tensor class.
 *
 * @tparam T The numeric type of the tensor elements.
 */
template <typename T> class TensorData
{
  public:
    /**
     * @brief Default constructor.
     */
    TensorData() = default;

    /**
     * @brief Copy constructor is deleted to avoid shallow copies of unique_ptr.
     */
    TensorData(const TensorData&) = delete;

    /**
     * @brief Move constructor.
     * @param other The source TensorData to move from.
     */
    TensorData(TensorData&& other) noexcept 
        : data_(std::move(other.data_)), 
          shape_(std::move(other.shape_)), 
          stride_(std::move(other.stride_)) {}

    /**
     * @brief Copy assignment is deleted.
     */
    TensorData& operator=(const TensorData&) = delete;

    /**
     * @brief Move assignment operator.
     * @param other The source TensorData to move from.
     * @return TensorData& reference to this.
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
     * @brief Destructor.
     */
    ~TensorData() = default;

    /**
     * @brief Construct from NestedData, moving the flattened data into managed memory.
     * @param data The source NestedData (typically from an initializer list).
     */
    // NOLINTNEXTLINE google-explicit-constructor
    TensorData(NestedData<T>&& data)
        : shape_(std::move(data.shape))
    {
        size_t size = data.flatData.size();
        data_ = std::make_unique<T[]>(size);
        std::move(data.flatData.begin(), data.flatData.end(), data_.get());
        stride_ = TensorStride(shape_);
    }

  private:
    std::unique_ptr<T[]> data_;
    TensorShape shape_;
    TensorStride stride_;

    friend class TensorDataTest;
    friend class Tensor<T>;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_DATA_H