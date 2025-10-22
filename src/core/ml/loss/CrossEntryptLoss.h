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

#ifndef HAHAHA_CROSSENTROPYLOSS_H
#define HAHAHA_CROSSENTROPYLOSS_H
#include "../../Tensor.h"
#include "Loss.h"
#include "core/defines/h3defs.h"

HHH_NAMESPACE_IMPORT

namespace hahaha::ml
{

template <typename T> class CrossEntropyLoss final : public Loss<T>
{
  public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() override = default;

    Tensor<T> forward(const Tensor<T>& input, const Tensor<T>& target) override
    {
        // Cross Entropy = -sum(target * log(input))
        // Assumes input is already probabilities (after softmax)
        // and target is one-hot encoded

        if (input.shape() != target.shape())
        {
            throw std::runtime_error(
                "Input and target shapes must match for Cross Entropy loss");
        }

        Tensor<T> result({1});
        T loss = static_cast<T>(0);

        // Compute -sum(target * log(input + epsilon)) where epsilon prevents
        // log(0)
        constexpr T epsilon = static_cast<T>(1e-7);

        for (sizeT i = 0; i < input.size(); ++i)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T input_val = input[i] + epsilon; // Avoid log(0)
                loss += target[i] * math::log(input_val);
            }
        }

        result.set({0}, -loss / static_cast<T>(input.size()));
        return result;
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_CROSSENTROPYLOSS_H
