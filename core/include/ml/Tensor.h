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
#include <memory>
#include <string>
#include <iostream>

#include "backend/Device.h"
#include "backend/Storage.h"

namespace h3::ml {

class TensorImpl;

using backend::ScalarType;
using backend::Device;
using backend::DeviceType;
using common::SizeType;

class Tensor {
public:
    Tensor();
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    // Properties
    [[nodiscard]] SizeType dim() const;
    [[nodiscard]] SizeType size(SizeType dim) const;
    [[nodiscard]] const std::vector<SizeType>& sizes() const;
    [[nodiscard]] SizeType numel() const;
    [[nodiscard]] ScalarType scalar_type() const;
    [[nodiscard]] Device device() const;
    [[nodiscard]] bool defined() const;

    // Operations (simple dispatch)
    [[nodiscard]] Tensor add(const Tensor& other, float alpha = 1.0f) const;
    [[nodiscard]] Tensor mul(const Tensor& other) const;
    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] Tensor relu() const;

    // In-place operations
    Tensor& add_(const Tensor& other, float alpha = 1.0f);
    Tensor& mul_(const Tensor& other);
    Tensor& relu_();

    // Backward (autograd placeholder)
    void backward();
    [[nodiscard]] Tensor grad() const;
    void set_grad(const Tensor& grad);
    [[nodiscard]] bool requires_grad() const;
    void set_requires_grad(bool requires_grad);

    // Print
    [[nodiscard]] std::string toString() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

// Implementation details (PIMPL)
class TensorImpl {
public:
    TensorImpl(backend::Storage storage,
               ScalarType dtype,
               std::vector<SizeType> sizes,
               std::vector<SizeType> strides);
    virtual ~TensorImpl() = default;

    backend::Storage storage_;
    ScalarType dtype_;
    std::vector<SizeType> sizes_;
    std::vector<SizeType> strides_;

    // Autograd
    Tensor grad_;
    bool requires_grad_ = false;
};

} // namespace h3