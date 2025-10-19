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
// Created by Napbad on 10/9/25.
//

#ifndef HAHAHA_LAYER_H
#define HAHAHA_LAYER_H

#include <ml/Tensor.h>

#include "compute/Variable.h"

namespace hahaha::ml
{
template <typename T> class Layer
{
  public:
    virtual ~Layer() = default;

    // Perform the forward pass
    virtual Variable<T> *forward(const Variable<T>* input) = 0;

    // Get the layer's parameters (weights, biases, etc.)
    virtual ds::Vector<Variable<T>*> parameters()
    {
        return {};
    }
};
} // namespace hahaha::ml

#endif // HAHAHA_LAYER_H
