// Copyright (c) 2025 Napbad
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
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
// Created: 2025-12-16 14:43:32 by Napbad
// Updated: 2025-12-21 13:48:39
// 


#ifndef HAHAHA_TENSOR_H
#define HAHAHA_TENSOR_H

#include "common/definitions.h"

namespace hahaha::math
{

    using namespace hahaha::common;

    /**
     * @brief Tensor 
     *
     * @tparam ValueType
     */
    template<typename ValueType>

class Tensor
{
  public:
    Tensor() = default;
    ~Tensor() = default;

    Tensor operator+(const Tensor& other)
    {
        Tensor result;
        
        return result;
    }

  private:
    ValueType* data_;
    u32 size_;
       
};

} // namespace hahaha::math

#endif // HAHAHA_TENSOR_H