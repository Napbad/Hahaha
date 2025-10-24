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
// Created by Napbad on 7/19/25.
//

#ifndef HAHAHA_LOSS_H
#define HAHAHA_LOSS_H
#include "../../Tensor.h"
#include "compute/Variable.h"
#include "core/defines/h3defs.h"

HHH_NAMESPACE_IMPORT

namespace hahaha::ml
{
template <typename T> class Loss
{
  public:
    virtual ~Loss() = default;
    virtual Variable<T>* forward(const Variable<T>& input,
                                 const Variable<T>& target) = 0;

    Variable<T>* operator()(const Variable<T>& input, const Variable<T>& target)
    {
        Variable<T> resultVar = forward(input, target);
        return new Variable<T>(resultVar);
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_LOSS_H
