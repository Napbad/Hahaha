// Copyright (c) 2025 Napbad
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
// Created: 2025-12-16 14:43:32 by Napbad
// Updated: 2025-12-23 06:47:06
//

#ifndef HAHAHA_MATH_DS_TENSOR_SHAPE_H
#define HAHAHA_MATH_DS_TENSOR_SHAPE_H

#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

#include "common/definitions.h"

namespace hahaha::math
{

using namespace hahaha::common;

/**
 * @brief TensorShape
 *
 */
class TensorShape
{
  public:
    TensorShape() = default;
    TensorShape(const TensorShape&) = default;
    TensorShape(TensorShape&&) = delete;
    TensorShape& operator=(const TensorShape&) = default;
    TensorShape& operator=(TensorShape&&) = delete;
    ~TensorShape() = default;

    TensorShape(const std::initializer_list<u32> dims)
    {
        dims_.reserve(dims.size());
        for (const auto& dim : dims)
        {
            dims_.emplace_back(dim);
        }
    }

    std::vector<u32> dims() const
    {
        return dims_;
    }

  private:
    std::vector<u32> dims_;

    friend class TensorShapeTest;
};

} // namespace hahaha::math

#endif // HAHAHA_MATH_DS_TENSOR_SHAPE_H