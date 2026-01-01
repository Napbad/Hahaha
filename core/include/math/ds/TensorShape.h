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

#ifndef HAHAHA_MATH_DS_TENSOR_SHAPE_H
#define HAHAHA_MATH_DS_TENSOR_SHAPE_H

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "common/definitions.h"
#include "utils/common/HelperStruct.h"

class TensorShapeTest;

namespace hahaha::math {

using hahaha::common::u32;
using hahaha::utils::isInitList;
using hahaha::utils::isLegalDataType;

/**
 * @brief Represents the shape (dimensions) of a tensor.
 *
 * TensorShape stores a vector of integers representing the size of each
 * dimension. For example, a 2x3 matrix has dimensions {2, 3}.
 *
 * This class provides utilities to compute the total number of elements
 * and format the shape as a string.
 */
class TensorShape {
  public:
    /** @brief Construct an empty shape. */
    TensorShape() = default;

    /** @brief Default destructor. */
    ~TensorShape() = default;

    /** @brief Copy-construct a TensorShape. */
    TensorShape(const TensorShape&) = default;

    /**
     * @brief Move-construct a TensorShape.
     * @param other Source to move from.
     */
    TensorShape(TensorShape&& other) noexcept : dims_(std::move(other.dims_)) {
    }

    /**
     * @brief Construct from an initializer list of dimensions.
     * @param dims List of dimension sizes (e.g., {2, 3, 4}).
     */
    TensorShape(const std::initializer_list<size_t> dims) : dims_(dims) {
    }

    /**
     * @brief Construct from a vector of dimensions.
     * @param dims Vector of dimension sizes.
     */
    explicit TensorShape(const std::vector<size_t>& dims) : dims_(dims) {
        if (dims_.empty()) {
            // Ensure that a shape with an empty vector explicitly represents a
            // 0-dimensional scalar.
            dims_.clear();
        }
    }

    /** @brief Copy assignment operator. */
    TensorShape& operator=(const TensorShape&) = default;

    /** @brief Move assignment operator. */
    TensorShape& operator=(TensorShape&& other) noexcept = default;

    /**
     * @brief Get the dimensions vector.
     * @return const std::vector<size_t>& dimensions.
     */
    [[nodiscard]] const std::vector<size_t>& getDims() const {
        return dims_;
    }

    /**
     * @brief Compute the total number of elements for this shape.
     *
     * Total size is the product of all dimension sizes.
     * Formula: size = product(dims[i]) for i in 0..N-1
     *
     * @return product of all dimensions (1 for empty shape).
     */
    [[nodiscard]] size_t getTotalSize() const {
        if (dims_.empty()) {
            return 1;
        }
        size_t size = 1;
#pragma unroll 5
        for (const auto& dim : dims_) {
            size *= dim;
        }
        return size;
    }

    /** @brief Reverse the dimensions (e.g., for converting layout). */
    void reverse() {
        std::reverse(dims_.begin(), dims_.end());
    }

    /**
     * @brief Return a string representation of the shape.
     * @return std::string formatted as "(d1, d2, ...)".
     */
    [[nodiscard]] std::string toString() const {
        std::string result = "(";
#pragma unroll 5
        for (size_t i = 0; i < dims_.size(); ++i) {
            result += std::to_string(dims_[i]);
            if (i != dims_.size() - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    /** @brief Equality operator. */
    bool operator==(const TensorShape& other) const {
        return dims_ == other.dims_;
    }

    /** @brief Inequality operator. */
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

  private:
    std::vector<size_t> dims_; /**< Vector of dimension sizes. */

    // Friend class for testing
    friend class ::TensorShapeTest;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_SHAPE_H
