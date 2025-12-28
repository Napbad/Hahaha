// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
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
// Created: 2025-12-14 22:59:42 by Napbad
// Updated: 2025-12-16 13:58:19
//

#ifndef HAHAHA_MATH_DS_TENSOR_DATA_H
#define HAHAHA_MATH_DS_TENSOR_DATA_H

#include <memory>

#include "math/ds/NestedData.h"
#include "math/ds/TensorShape.h"
#include "math/ds/TensorStride.h"
#include "utils/common/HelperStruct.h"

namespace hahaha::math
{

template <typename T> class Tensor;

using hahaha::common::u32;
using hahaha::util::isInitList;
using hahaha::util::isNestedInitList;

/**
 * @brief TensorData
 *
 * @tparam ValueType
 */
template <typename T> class TensorData
{
  public:
    TensorData() = default;
    TensorData(const TensorData&) = delete;
    TensorData(TensorData&&) = delete;
    TensorData& operator=(const TensorData&) = delete;
    TensorData& operator=(TensorData&&) = delete;
    ~TensorData() = default;

    // NOLINTNEXTLINE google-explicit-constructor
    TensorData(const NestedData<T> data)
        : data_(std::make_unique<T[]>(data.flatData.size())),
          shape_(std::move(data.shape))
    {
        std::copy(data.flatData.begin(), data.flatData.end(), data_.get());
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