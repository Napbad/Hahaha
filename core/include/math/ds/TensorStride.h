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

#ifndef HAHAHA_MATH_DS_TENSOR_STRIDE_H
#define HAHAHA_MATH_DS_TENSOR_STRIDE_H

#include <cstdio>
#include <sstream>
#include <vector>

#include "common/definitions.h"
#include "math/ds/TensorShape.h"

namespace hahaha::math
{
using hahaha::common::u32;

/**
 * @brief Represents the memory strides of a tensor.
 *
 * TensorStride stores a vector of unsigned 32-bit integers representing the
 * number of elements to skip to move to the next element in each dimension.
 * For a row-major tensor, the last dimension has a stride of 1.
 */
class TensorStride
{

  public:
    /**
     * @brief Default constructor for an empty TensorStride.
     */
    TensorStride() = default;

    /**
     * @brief Construct strides from a vector of dimensions.
     * @tparam T The numeric type of dimensions.
     * @param dims Vector of dimension sizes.
     */
    template <typename T> explicit TensorStride(const std::vector<T>& dims)
    {
        strides_.resize(dims.size());
        if (dims.empty())
        {
            return;
        }
        strides_[dims.size() - 1] = 1;
#pragma unroll 5
        for (size_t i = dims.size() - 1; i > 0; --i)
        {
            strides_[i - 1] = strides_[i] * static_cast<u32>(dims[i]);
        }
    }

    /**
     * @brief Construct strides from a TensorShape.
     * @param shape The source shape to compute strides for.
     */
    explicit TensorStride(const TensorShape& shape) : TensorStride(shape.dims())
        {
    }

    /**
     * @brief Return a reference to the strides vector.
     * @return const std::vector<u32>& strides.
     */
    [[nodiscard]] const std::vector<u32>& dims() const
    {
        return strides_;
    }

    /**
     * @brief Access stride by dimension index.
     * @param index Dimension index.
     * @return u32 The stride value at that dimension.
     */
    [[nodiscard]] u32 operator[](u32 index) const
    {
        return strides_[index];
    }

    /**
     * @brief Return the number of dimensions.
     * @return u32 dimension count.
     */
    [[nodiscard]] u32 size() const
    {
        return strides_.size();
    }

    /**
     * @brief Return a string representation of the strides.
     * @return std::string formatted as "[s1, s2, ...]".
     */
    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstream;
        sstream << "[";

#pragma unroll 5
        for (u32 i = 0; i < strides_.size(); ++i)
        {
            sstream << strides_[i];
            if (i != strides_.size() - 1)
            {
                sstream << ", ";
            }
        }
        sstream << "]";
        return sstream.str();
    }

    /**
     * @brief Reverse the order of strides (e.g. for switching between row/column major).
     */
    void reverse()
    {
        std::reverse(strides_.begin(), strides_.end());
    }

  private:
    std::vector<u32> strides_;
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_STRIDE_H
