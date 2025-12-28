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

// Created: 2025-12-28 01:56:49 by Napbad
//

#ifndef HAHAHA_MATH_TENSOR_H
#define HAHAHA_MATH_TENSOR_H

#include <vector>
#include "math/ds/TensorData.h"

class TensorDataTest;

namespace hahaha::math
{

template <typename T> class Tensor
{
  public:
    Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) = delete;
    ~Tensor() = default;

    // NOLINTNEXTLINE google-explicit-constructor
    Tensor(NestedData<T> data) : data_(std::move(data))
    {

    }

    std::unique_ptr<T[]>& getRawData()
    {
        return data_.data_;
    }

    [[nodiscard]] TensorShape shape() const
    {
        return data_.shape_;
    }

    [[nodiscard]] TensorStride stride() const
    {
        return data_.stride_;
    }

  private:
    TensorData<T> data_;

    // used for autograd and compute graph
    std::vector<TensorData<T>> forwardTensor_;

    // default name of the test fixture class to this class
    friend class ::TensorDataTest;

};
} // namespace hahaha::math

#endif // HAHAHA_MATH_TENSOR_H