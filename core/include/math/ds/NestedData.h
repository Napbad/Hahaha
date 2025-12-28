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

// Created: 2025-12-28 01:31:07 by Napbad
//

#ifndef HAHAHA_MATH_DS_NESTED_DATA_H
#define HAHAHA_MATH_DS_NESTED_DATA_H

#include <initializer_list>
#include <vector>

#include "common/definitions.h"

// Forward declaration for friend class template
class NestedDataTest;

namespace hahaha::math
{

template <typename T> class TensorData;

using hahaha::common::u32;

template <typename T> struct NestedData
{

  public:
    // NOLINTNEXTLINE google-explicit-constructor
    NestedData(T data)
    {
        flatData.push_back(data);
    }

    NestedData(std::initializer_list<NestedData<T>> data)
    {
        shape.push_back(static_cast<u32>(data.size()));
        shape.insert(shape.end(),
                     data.begin()->shape.begin(),
                     data.begin()->shape.end());

#pragma unroll 5
        for (const auto& val : data)
        {
            flatData.insert(
                flatData.end(), val.flatData.begin(), val.flatData.end());
        }
    }

    // Public getter methods to access private members for testing
    const std::vector<T>& getFlatData() const
    {
        return flatData;
    }
    [[nodiscard]] const std::vector<u32>& getShape() const
    {
        return shape;
    }

  private:
    std::vector<T> flatData;
    std::vector<u32> shape;

    friend class TensorData<T>;
    friend class ::NestedDataTest;
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_NESTED_DATA_H