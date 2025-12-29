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

/**
 * @brief Utility structure for handling nested initializer lists to initialize tensors.
 *
 * NestedData recursively flattens multi-dimensional initializer lists into a 1D vector
 * while keeping track of the tensor's shape.
 *
 * @tparam T The numeric type of the tensor data.
 */
template <typename T> struct NestedData
{

  public:
    /**
     * @brief Construct from a single scalar value.
     * @param data The scalar value.
     */
    // NOLINTNEXTLINE google-explicit-constructor
    NestedData(T data)
    {
        flatData.push_back(data);
    }

    /**
     * @brief Construct from a nested initializer list.
     * @param data The nested initializer list.
     */
    NestedData(std::initializer_list<NestedData<T>> data)
    {
        if (data.size() == 0) { return;
}

        shape.push_back(static_cast<u32>(data.size()));
        const auto& firstShape = data.begin()->shape;
        shape.insert(shape.end(), firstShape.begin(), firstShape.end());

        size_t totalElements = 0;
        #pragma unroll 5
        for (const auto& val : data) {
            totalElements += val.flatData.size();
        }
        flatData.reserve(totalElements);

#pragma unroll 5
        for (const auto& val : data)
        {
            flatData.insert(
                flatData.end(), val.flatData.begin(), val.flatData.end());
        }
    }

    /**
     * @brief Get the flattened 1D data.
     * @return const std::vector<T>& reference to flat data.
     */
    const std::vector<T>& getFlatData() const
    {
        return flatData;
    }

    /**
     * @brief Get the computed shape of the nested data.
     * @return const std::vector<u32>& reference to shape dimensions.
     */
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