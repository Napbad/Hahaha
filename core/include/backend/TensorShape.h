// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#pragma once

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <optional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

#include "common/data/Types.h"

namespace h3::backend {

/**
 * @brief Represents the shape (dimensions) of a tensor.
 *
 * Size stores a vector of integers representing the size of each
 * dimension. For example, a 2x3 matrix has dimensions {2, 3}.
 */
class TensorShape {

    using size_type = common::Int64;

public:
    TensorShape() = default;
    ~TensorShape() = default;
    TensorShape(const TensorShape&) = default;
    TensorShape(TensorShape&&) noexcept = default;
    TensorShape& operator=(const TensorShape&) = default;
    TensorShape& operator=(TensorShape&&) noexcept = default;

    TensorShape(const std::initializer_list<size_type> dims) : dims_(dims) {
    }

    explicit TensorShape(const std::vector<size_type>& dims) : dims_(dims) {
    }

    [[nodiscard]] const std::vector<size_type>& vec() const {
        return dims_;
    }

    [[nodiscard]] std::vector<size_type>& vec() {
        return dims_;
    }

    [[nodiscard]] size_type numel() const {
        if (dims_.empty())
            return 1;
        // Scalar or empty? PyTorch convention: empty size -> scalar?
        // No, empty vec usually means scalar in some contexts,
        // but here let's follow c10::IntArrayRef or similar.
        // Actually, for a tensor with shape {}, it is a scalar, numel is 1.
        // For a tensor with shape {0}, it is empty, numel is 0.
        size_type size = 1;
        for (const auto& dim : dims_) {
            size *= dim;
        }
        return size;
    }

    [[nodiscard]] size_t rank() const {
        return dims_.size();
    }

    [[nodiscard]] size_type operator[](size_t index) const {
        return dims_[index];
    }

    [[nodiscard]] size_type& operator[](size_t index) {
        return dims_[index];
    }

    bool operator==(const TensorShape& other) const {
        return dims_ == other.dims_;
    }

    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

    [[nodiscard]] std::string toString() const {
        std::string result = "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            result += std::to_string(dims_[i]);
            if (i != dims_.size() - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    // Broadcast utility
    static std::optional<TensorShape> broadcast(const TensorShape& a, const TensorShape& b) {
        std::vector<size_type> result_dims;
        size_type i = a.rank() - 1;
        size_type j = b.rank() - 1;

        while (i >= 0 || j >= 0) {
            size_type dim_a = i >= 0 ? a[i] : 1;

            if (size_type dim_b = (j >= 0) ? b[j] : 1; dim_a == dim_b || dim_a == 1 ||
                dim_b == 1) {
                result_dims.push_back(std::max(dim_a, dim_b));
            } else {
                return std::nullopt;
            }
            i--;
            j--;
        }
        std::ranges::reverse(result_dims);
        return TensorShape(result_dims);
    }

private:
    std::vector<size_type> dims_;
};

}