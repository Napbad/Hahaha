// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include <stdexcept>
#include <initializer_list>
#include <vector>

// Forward declaration for friend class template
class NestedDataTest;

namespace hahaha::math {

template <typename T> class TensorData;

/**
 * @brief Utility structure for handling nested initializer lists to initialize
 * tensors.
 *
 * NestedData recursively flattens multi-dimensional initializer lists into a 1D
 * vector while keeping track of the tensor's shape.
 *
 * Example:
 *   NestedData<float> data = {{1, 2}, {3, 4}};
 *   // flatData_ becomes {1, 2, 3, 4}, shape_ becomes {2, 2}
 *
 * @tparam T The numeric type of the tensor data.
 */
template <typename T> struct NestedData {

  public:
    /**
     * @brief Construct from a single scalar value.
     * @param data The scalar value.
     */
    // NOLINTNEXTLINE google-explicit-constructor
    NestedData(T data) : shape_({}) {
        flatData_.push_back(data);
    }

    /**
     * @brief Construct from a nested initializer list.
     *
     * This constructor recursively flattens the list and computes the shape.
     *
     * @param data The nested initializer list.
     */
    NestedData(std::initializer_list<NestedData<T>> data) {
        if (data.size() == 0) {
            return;
        }

        shape_.push_back(data.size());
        const auto& firstNestedData = *data.begin();
        // Check if all nested data have the same shape
        for (const auto& val : data) {
            if (val.shape_ != firstNestedData.shape_) {
                throw std::invalid_argument("Nested initializer list has "
                                            "inconsistent shapes.");
            }
        }
        shape_.insert(shape_.end(), firstNestedData.shape_.begin(),
                      firstNestedData.shape_.end());

        size_t totalElements = 0;
#pragma unroll 5
        for (const auto& val : data) {
            totalElements += val.flatData_.size();
        }
        flatData_.reserve(totalElements);

#pragma unroll 5
        for (const auto& val : data) {
            flatData_.insert(
                flatData_.end(), val.flatData_.begin(), val.flatData_.end());
        }
    }

    /**
     * @brief Get the flattened 1D data.
     * @return const std::vector<T>& reference to flat data.
     */
    const std::vector<T>& getFlatData() const {
        return flatData_;
    }

    /**
     * @brief Get the computed shape of the nested data.
     * @return const std::vector<size_t>& reference to shape dimensions.
     */
    [[nodiscard]] const std::vector<size_t>& getShape() const {
        return shape_;
    }

  private:
    std::vector<T> flatData_;   /**< Linearized multi-dimensional data. */
    std::vector<size_t> shape_; /**< Computed shape of the input list. */

    friend class TensorData<T>;
    friend class ::NestedDataTest;
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_NESTED_DATA_H
