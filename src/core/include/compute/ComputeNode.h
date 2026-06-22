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

#ifndef HAHAHA_COMPUTENODE_H_D3F0BB4E059144318EA7871C7175E12A
#define HAHAHA_COMPUTENODE_H_D3F0BB4E059144318EA7871C7175E12A
#include "math/TensorInner.h"
#include "utils/OwnPointer.h"

namespace h3::core::compute {
class ComputeNode {
public:
    ComputeNode() {
        m_tensor = nullptr;
    }
    explicit ComputeNode(const math::TensorShape& shape)
        : m_tensor(utils::make_own_ptr<math::TensorInner>(shape)) {
    }

    explicit ComputeNode(const utils::OwnPointer<math::TensorInner>& ownerPointer) {
        m_tensor = ownerPointer;
    }

    explicit ComputeNode(const math::TensorInner&& tensorInner) : m_tensor(
        utils::make_own_ptr<math::TensorInner>(tensorInner)) {
    }

    utils::OwnPointer<math::TensorInner> tensorInner() {
        return m_tensor;
    }

    [[nodiscard]] utils::OwnPointer<math::TensorInner> tensorInner() const {
        return m_tensor;
    }

    [[nodiscard]] backend::Device device() const {
        return m_tensor->device();
    }
    [[nodiscard]] ComputeNode add(const ComputeNode& other) const;
    ComputeNode sub(const ComputeNode& other) const;
    ComputeNode mul(const ComputeNode& other) const;
    ComputeNode div(const ComputeNode& other) const;

    ComputeNode& add_(const ComputeNode& other);
    ComputeNode& sub_(const ComputeNode& other);
    ComputeNode& mul_(const ComputeNode& other);
    ComputeNode& div_(const ComputeNode& other);

    ComputeNode operator +(const ComputeNode& other) const;
    ComputeNode operator -(const ComputeNode& other) const;
    ComputeNode operator *(const ComputeNode& other) const;
    ComputeNode operator /(const ComputeNode& other) const;

    [[nodiscard]] ComputeNode matmul(const ComputeNode& other) const;

    ComputeNode to(backend::Device device);
    ComputeNode to(backend::Device device, DataType dtype);

    [[nodiscard]] ComputeNode view() const;
    [[nodiscard]] ComputeNode broadcastView(const math::TensorShape& newShape) const;
    [[nodiscard]] ComputeNode reshape(const math::TensorShape& newShape) const;
    [[nodiscard]] ComputeNode transpose(int64_t dim0, int64_t dim1) const;
    [[nodiscard]] ComputeNode permute(const std::vector<int64_t>& dims) const;
    [[nodiscard]] ComputeNode squeeze(int64_t dim = -1) const;
    [[nodiscard]] ComputeNode unsqueeze(int64_t dim) const;

    void setTensorInner(math::TensorInner&& tensor_inner);

private:
    utils::OwnPointer<math::TensorInner> m_tensor;
};
} // namespace h3::core::compute

#endif // HAHAHA_COMPUTENODE_H_D3F0BB4E059144318EA7871C7175E12A