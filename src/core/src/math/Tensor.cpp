//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
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

#include "ml/Tensor.h"

#include "utils/handler/exception_handler.h"

namespace h3::core::ml {

Tensor Tensor::add(const Tensor& other) const {
    return Tensor(m_node + other.m_node);
}

Tensor Tensor::sub(const Tensor& other) const {
    return Tensor(m_node - other.m_node);
}

Tensor Tensor::mul(const Tensor& other) const {
    return Tensor(m_node * other.m_node);
}

Tensor Tensor::div(const Tensor& other) const {
    return Tensor(m_node / other.m_node);
}

Tensor Tensor::operator+(const Tensor& other) const {
    return add(other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return sub(other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return mul(other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return div(other);
}

Tensor& Tensor::add_(const Tensor& other) {
    m_node = m_node + other.m_node;
    return *this;
}

Tensor& Tensor::sub_(const Tensor& other) {
    m_node = m_node - other.m_node;
    return *this;
}

Tensor& Tensor::mul_(const Tensor& other) {
    m_node = m_node * other.m_node;
    return *this;
}

Tensor& Tensor::div_(const Tensor& other) {
    m_node = m_node / other.m_node;
    return *this;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // Stub implementation - requires actual matrix multiplication kernel
    (void)other;
    ThrowInvalid("matmul not implemented yet");
    return *this;
}

Tensor Tensor::to(backend::Device device) {
    (void)device;
    ThrowInvalid("to() not implemented yet");
    return *this;
}

Tensor Tensor::to(backend::Device device, DataType dtype) {
    (void)device;
    (void)dtype;
    ThrowInvalid("to() not implemented yet");
    return *this;
}

Tensor Tensor::view(const math::TensorShape& new_shape) const {
    (void)new_shape;
    ThrowInvalid("view not implemented yet");
    return *this;
}

Tensor Tensor::reshape(const math::TensorShape& new_shape) const {
    (void)new_shape;
    ThrowInvalid("reshape not implemented yet");
    return *this;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    (void)dim0;
    (void)dim1;
    ThrowInvalid("transpose not implemented yet");
    return *this;
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    (void)dims;
    ThrowInvalid("permute not implemented yet");
    return *this;
}

Tensor Tensor::squeeze(int64_t dim) const {
    (void)dim;
    ThrowInvalid("squeeze not implemented yet");
    return *this;
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    (void)dim;
    ThrowInvalid("unsqueeze not implemented yet");
    return *this;
}

} // namespace h3::core::ml
