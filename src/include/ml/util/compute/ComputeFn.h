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

#ifndef HAHAHA_COMPUTEFN_H
#define HAHAHA_COMPUTEFN_H
#include "common/util/vectorize/vectorize.h"
#include "ml/common/Tensor.h"

namespace hahaha::ml::util {
    // define some common functions for compute graph usage
    class ComputeErr final : public BaseErr {};

    class ComputeFn {
    public:
        template <typename T>
        Res<Tensor<T>, ComputeErr> add(Tensor<T> src1, Tensor<T> src2) {
            SetRetT(Tensor<T>, ComputeErr) if (src1.shape() != src2.shape()){
                Err("the src1's shape is not same to the src2!")} Ok(src1 + src2);
        }

        template <typename T>
        Res<Tensor<T>, ComputeErr> mul(Tensor<T> src1, Tensor<T> src2) {
            SetRetT(Tensor<T>, ComputeErr) if (src1.shape() != src2.shape()){
                Err("the src1's shape is not same to the src2!")} Ok(src1 * src2);
        }

        template <typename T>
        Res<Tensor<T>, ComputeErr> sub(Tensor<T> src1, Tensor<T> src2) {
            SetRetT(Tensor<T>, ComputeErr) if (src1.shape() != src2.shape()){
                Err("the src1's shape is not same to the src2!")} Ok(src1 - src2);
        }
    };


} // namespace hahaha::ml::util

#endif // HAHAHA_COMPUTEFN_H
