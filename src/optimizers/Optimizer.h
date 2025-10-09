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

#ifndef HAHAHA_OPTIMIZER_H
#define HAHAHA_OPTIMIZER_H
#include <vector>

#include "defines/h3defs.h"
#include "ds/Vector.h"
#include "compute/Variable.h"

HHH_NAMESPACE_IMPORT

    namespace hahaha::ml
{
    template <typename T> class Optimizer
    {

      public:
        virtual ~Optimizer() = default;
        Optimizer(const ds::Vector<Variable<T>>& parameters, const f64 learningRate)
            : _parameters(parameters), _learningRate(learningRate)
        {
        }
        virtual void step() = 0;
        virtual void zero_grad()
        {
            for (auto& param : _parameters)
            {
                param.zeroGrad();
            }
        }

      private:
        ds::Vector<Variable<T>> _parameters{};
        f64 _learningRate;
    };
}

#endif // HAHAHA_OPTIMIZER_H
