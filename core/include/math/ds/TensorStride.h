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
// Created: 2025-12-16 14:43:42 by Napbad
// Updated: 2025-12-26 06:53:20
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

class TensorStride
{

  public:
    TensorStride() = default;

    template <typename T> explicit TensorStride(const std::vector<T>& dims)
    {
        strides_.resize(dims.size());
        if (dims.size() == 0)
        {
            return;
        }
        strides_[dims.size() - 1] = 1;
#pragma unroll 5
        for (u32 i = dims.size() - 2; i >= 0; --i)
        {
            strides_[i] = strides_[i + 1] * static_cast<u32>(dims[i + 1]);
        }
    }

    explicit TensorStride(const TensorShape& shape)
    {
        const auto dims = shape.dims();
        strides_.resize(dims.size());
        if (dims.size() == 0)
        {
            return;
        }
        strides_[dims.size() - 1] = 1;
        if (dims.size() > 2)
        {
#pragma unroll 5
            for (u32 i = dims.size() - 2; i > 0; --i)
            {
                strides_[i] = strides_[i + 1] * static_cast<u32>(dims[i + 1]);
            }
        }
        strides_[0] = strides_[1] * static_cast<u32>(dims[1]);
    }

    [[nodiscard]] const std::vector<u32>& dims() const
    {
        return strides_;
    }

    [[nodiscard]] u32 operator[](u32 index) const
    {
        return strides_[index];
    }

    [[nodiscard]] u32 size() const
    {
        return strides_.size();
    }

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

    void reverse()
    {
        std::reverse(strides_.begin(), strides_.end());
    }

  private:
    std::vector<u32> strides_;
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_STRIDE_H
