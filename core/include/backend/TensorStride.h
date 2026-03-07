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
#include <sstream>
#include <stdexcept>

#include "TensorShape.h"
#include "common/data/Types.h"

namespace h3::backend {

/**
 * @brief Represents the memory strides of a tensor.
 */
class TensorStride {
    using size_type = common::Int64;
public:
    TensorStride() = default;

    // Compute contiguous strides from shape
    explicit TensorStride(const TensorShape& shape) {
        if (shape.rank() == 0)
            return;
        strides_.resize(shape.rank());
        size_type stride = 1;
        for (size_t i = shape.rank(); i > 0; --i) {
            strides_[i - 1] = stride;
            stride *= shape[i - 1];
        }
    }

    explicit TensorStride(const std::vector<size_type>& strides) : strides_(strides) {
    }

    [[nodiscard]] const std::vector<size_type>& vec() const {
        return strides_;
    }

    [[nodiscard]] size_type operator[](const size_t index) const {
        return strides_[index];
    }

    [[nodiscard]] std::string toString() const {
        std::stringstream sstream;
        sstream << "[";
        for (size_t i = 0; i < strides_.size(); ++i) {
            sstream << strides_[i];
            if (i != strides_.size() - 1) {
                sstream << ", ";
            }
        }
        sstream << "]";
        return sstream.str();
    }

private:
    std::vector<size_type> strides_;
};

}