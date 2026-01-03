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

#ifndef HAHAHA_MATH_DS_TENSOR_STRIDE_H
#define HAHAHA_MATH_DS_TENSOR_STRIDE_H

#include <sstream>
#include <vector>

#include "common/definitions.h"
#include "math/ds/TensorShape.h"

namespace hahaha::math {
using common::u32;

/**
 * @brief Represents the memory strides of a tensor.
 *
 * Strides define how many elements to skip in memory to move to the next
 * element in a specific dimension.
 *
 * For a row-major tensor of shape (d0, d1, ..., dN):
 * - stride[N] = 1
 * - stride[i] = stride[i+1] * dims[i+1]
 */
class TensorStride {

  public:
    /** @brief Default constructor for an empty TensorStride. */
    TensorStride() = default;

    /**
     * @brief Construct strides from a vector of dimensions.
     * @tparam T The numeric type of dimensions.
     * @param dims Vector of dimension sizes.
     */
    template <typename T> explicit TensorStride(const std::vector<T>& dims) {
        strides_.resize(dims.size());
        if (dims.empty()) {
            return;
        }
        strides_[dims.size() - 1] = 1;
#pragma unroll 5
        for (size_t i = dims.size() - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * static_cast<size_t>(dims[i]);
        }
    }

    /**
     * @brief Construct strides from a TensorShape.
     * @param shape The source shape to compute strides for.
     */
    explicit TensorStride(const TensorShape& shape)
        : TensorStride(shape.getDims()) {
    }

    /**
     * @brief Get the strides vector.
     * @return const std::vector<size_t>& strides.
     */
    [[nodiscard]] const std::vector<size_t>& getStrides() const {
        return strides_;
    }

    [[nodiscard]] std::vector<size_t>& getStrides() {
        return strides_;
    }

    /**
     * @brief Access stride by dimension index.
     * @param index Dimension index.
     * @return size_t The stride value at that dimension.
     */
    [[nodiscard]] size_t operator[](size_t index) const {
        return strides_[index];
    }

    /**
     * @brief Return the number of dimensions.
     * @return size_t dimension count.
     */
    [[nodiscard]] size_t getSize() const {
        return strides_.size();
    }

    /**
     * @brief Return a string representation of the strides.
     * @return std::string formatted as "[s1, s2, ...]".
     */
    [[nodiscard]] std::string toString() const {
        std::stringstream sstream;
        sstream << "[";

#pragma unroll 5
        for (size_t i = 0; i < strides_.size(); ++i) {
            sstream << strides_[i];
            if (i != strides_.size() - 1) {
                sstream << ", ";
            }
        }
        sstream << "]";
        return sstream.str();
    }

    /**
     * @brief Reverse the order of strides (e.g., for switching layout).
     */
    void reverse() {
        std::reverse(strides_.begin(), strides_.end());
    }

    /**
     * @brief Access stride by dimension index.
     * @param index Dimension index.
     * @return size_t The stride value at that dimension.
     */
    size_t operator[](size_t index) {
        return strides_[index];
    }

    /**
     * @brief Access stride with bounds checking.
     * @param index Dimension index.
     * @return size_t The stride value.
     * @throw std::out_of_range If index is invalid.
     */
    size_t at(const size_t index) {
        if (index >= strides_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return strides_[index];
    }

    /**
     * @brief Const access stride with bounds checking.
     * @param index Dimension index.
     * @return size_t The stride value.
     * @throw std::out_of_range If index is invalid.
     */
    [[nodiscard]] size_t at(const size_t index) const {
        if (index >= strides_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return strides_[index];
    }

  private:
    std::vector<size_t> strides_; /**< Vector of computed stride values. */
};
} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_STRIDE_H
