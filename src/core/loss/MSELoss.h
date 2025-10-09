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
#include "Loss.h"
#include "common/defines/h3defs.h"
#include "ml/common/Tensor.h"

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{

    template <typename T> class MSELoss final : public Loss<T>
    {
      public:
        MSELoss() = default;
        ~MSELoss() override = default;

        Tensor<T> forward(const Tensor<T>& input, const Tensor<T>& target) override
        {
            // MSE = mean((input - target)^2)
            if (input.shape() != target.shape())
            {
                throw std::runtime_error("Input and target shapes must match for MSE loss");
            }

            auto diff = input - target;
            auto squared = diff * diff;

            // Return mean as a scalar tensor
            T mean_value = squared.sum() / static_cast<T>(squared.size());
            Tensor<T> result({1});
            result.set({0}, mean_value);

            return result;
        }
    };

} // namespace hahaha::ml

#endif // HAHAHA_MSELOSS_H
