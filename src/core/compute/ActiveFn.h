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

#ifndef HAHAHA_ACTIVEFN_H
#define HAHAHA_ACTIVEFN_H
#include <cmath>

#include "core/defines/h3defs.h"
#include "core/ml/Tensor.h"

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{

    class ActiveFn
    {
      public:
        static f32 sigmoid(const f32 x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        static f32 relu(const f32 x)
        {
            return x > 0.0f ? x : 0.0f;
        }

        static f32 tanh(const f32 x)
        {
            return std::tanh(x);
        }

        static f32 linear(const f32 x)
        {
            return x;
        }

        static Tensor<f32> softmax(const Tensor<f32>& x)
        {
            Tensor<f32> result(x.shape());
            result.fill(0);
            for (sizeT i = 0; i < x.size(); ++i)
            {
                result[i] = std::exp(x[i]);
            }
            return result / result.sum();
        }
    };
}

#endif // HAHAHA_ACTIVEFN_H
