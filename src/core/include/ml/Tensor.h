//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/26/26.
//

#ifndef HAHAHA_TENSOR_H_2ACEBAFB3C1048578DCC0E9ABDCA952B
#define HAHAHA_TENSOR_H_2ACEBAFB3C1048578DCC0E9ABDCA952B
#include <utility>

#include "compute/ComputeNode.h"

namespace h3::core::ml {
class Tensor {

  public:
    explicit Tensor(const math::TensorShape& shape) : m_node(shape) {
    }

    explicit Tensor(compute::ComputeNode node) : m_node(std::move(node)) {
    }

    [[nodiscard]] Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;

    Tensor& add_(const Tensor& other);
    Tensor& sub_(const Tensor& other);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    [[nodiscard]] Tensor matmul(const Tensor& other) const;

    Tensor to(backend::Device device);
    Tensor to(backend::Device device, DataType dtype);

    [[nodiscard]] Tensor view(const math::TensorShape& new_shape) const;
    [[nodiscard]] Tensor reshape(const math::TensorShape& new_shape) const;
    [[nodiscard]] Tensor transpose(int64_t dim0, int64_t dim1) const;
    [[nodiscard]] Tensor permute(const std::vector<int64_t>& dims) const;
    [[nodiscard]] Tensor squeeze(int64_t dim = -1) const;
    [[nodiscard]] Tensor unsqueeze(int64_t dim) const;

  private:
    compute::ComputeNode m_node;
};
} // namespace h3::core::ml

#endif // HAHAHA_TENSOR_H_2ACEBAFB3C1048578DCC0E9ABDCA952B
