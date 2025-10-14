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

#ifndef HAHAHA_AUTOGRADOPS_H
#define HAHAHA_AUTOGRADOPS_H
#include "Variable.h"
#include "core/defines/h3defs.h"

HHH_NAMESPACE_IMPORT

namespace hahaha::ml
{
template <typename T> class AutogradOps
{
  public:
    static Variable<T> add(const Variable<T>& a, const Variable<T>& b)
    {
        return a + b;
    }

    static Variable<T> mul(const Variable<T>& a, const Variable<T>& b)
    {
        return a * b;
    }

    static Variable<T> matmul(const Variable<T>& a, const Variable<T>& b)
    {
        return a.matmul(b);
    }

    static Variable<T> relu(const Variable<T>& a)
    {
        return a.relu();
    }

    static Variable<T> sigmoid(const Variable<T>& a)
    {
        return a.sigmoid();
    }
};

} // namespace hahaha::ml

#endif // HAHAHA_AUTOGRADOPS_H
