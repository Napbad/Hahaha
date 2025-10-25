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
// Created by napbad on 9/28/25.
//

#ifndef HAHAHA_COMMON_UTIL_VECTORIZE_H
#define HAHAHA_COMMON_UTIL_VECTORIZE_H
#include "../../core/TensorData.h"
namespace hahaha::core::util
{

using ml::TensorData;

class VectorizeErr final : public BaseErr
{
  public:
    VectorizeErr() = default;
    explicit VectorizeErr(const char* msg) : BaseErr(msg)
    {
    }
};

class Vectorize
{

  public:
    template <typename T>
    static Res<TensorData<T>, VectorizeErr>
    add(std::initializer_list<TensorData<T>> list)
    {
        SetRetT(TensorData<T>, VectorizeErr) auto shape = list.begin()->shape();
        for (auto& tensor : list)
        {
            if (shape != tensor.shape())
            {
                Err("Wrong tensor type during calc!")
            }
        }

        auto res = TensorData<T>(shape);
        res.fill(0);
        auto tmp = res;
        for (auto& val : list)
        {
            res += val;
        }
        Ok(res)
    }

    template <typename T>
    Res<TensorData<T>, VectorizeErr> sub(std::initializer_list<TensorData<T>> list)
    {
        SetRetT(TensorData<T>, VectorizeErr) auto shape = list.begin()->shape();
        for (auto& tensor : list)
        {
            if (shape != tensor.shape())
            {
                Err("Wrong tensor type during calc!")
            }
        }

        auto res = TensorData<T>(shape);
        res.copy(*list.begin());
        auto begin = list.begin();
        ++begin;
        for (; begin != list.end(); ++begin)
        {
            res -= *begin;
        }
        Ok(res)
    }

    template <typename T>
    static Res<TensorData<T>, VectorizeErr>
    mul(std::initializer_list<TensorData<T>> list)
    {
        SetRetT(TensorData<T>, VectorizeErr) auto shape = list.begin()->shape();
        for (auto& tensor : list)
        {
            if (shape != tensor.shape())
            {
                Err("Wrong tensor type during calc!")
            }
        }

        auto res = TensorData<T>(shape);
        res.fill(1); // Start with 1 for multiplication
        for (auto& val : list)
        {
            res *= val;
        }
        Ok(res)
    }

    template <typename T>
    static Res<TensorData<T>, VectorizeErr>
    div(std::initializer_list<TensorData<T>> list)
    {
        SetRetT(TensorData<T>, VectorizeErr) auto shape = list.begin()->shape();
        for (auto& tensor : list)
        {
            if (shape != tensor.shape())
            {
                Err("Wrong tensor type during calc!")
            }
        }

        auto res = TensorData<T>(shape);
        res.copy(*list.begin());
        auto begin = list.begin();
        ++begin;
        for (; begin != list.end(); ++begin)
        {
            res /= *begin;
        }
        Ok(res)
    }
};

} // namespace hahaha::core::util

#endif // HAHAHA_COMMON_UTIL_VECTORIZE_H
