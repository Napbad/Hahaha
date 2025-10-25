// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
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
// Created by napbad on 10/6/25.
//

#ifndef HAHAHA_MSELOSS_H
#define HAHAHA_MSELOSS_H
#include "../../TensorData.h"
#include "Loss.h"
#include "core/defines/h3defs.h"

HHH_NAMESPACE_IMPORT

namespace hahaha::ml
{

template <typename T> class MSELoss final : public Loss<T>
{
  public:
    MSELoss() = default;
    ~MSELoss() override = default;

    Variable<T>* forward(const Variable<T>& input,
                         const Variable<T>& target) override
    {
        // MSE = mean((input - target)^2)
        if (input.shape() != target.shape())
        {
            throw std::runtime_error(
                "Input and target shapes must match for MSE loss");
        }

        auto diff = input - target;
        auto squared = diff * diff;

        // Return mean as a scalar variable
        return new Variable<T>(squared.mean());
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_MSELOSS_H
