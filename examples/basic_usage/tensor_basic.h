// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#pragma once

#include <cstdio>

#include "Tensor.h"
#include "common/definitions.h"

void tensor_basic_init_example() {
    hahaha::Tensor<hahaha::common::u32> t1({1, 2});

    hahaha::Tensor<hahaha::common::u32> t2({{1, 2}, {3, 4}});

    hahaha::Tensor<hahaha::common::i32> t3(
        {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
}

void tensor_basic_add_example() {

    hahaha::Tensor<hahaha::common::u32> t1 = {{1, 2}};

    hahaha::Tensor<hahaha::common::u32> t2({3, 4});
    auto t3 = t1 + t2;
}

void tensor_basic_sub_example() {

    hahaha::Tensor<hahaha::common::u32> t1 = {{1, 2}};

    hahaha::Tensor<hahaha::common::u32> t2({3, 4});
    auto t3 = t1 - t2;
}

void tensor_basic_mul_example() {
    hahaha::Tensor<hahaha::common::u32> t1 = {{1, 2}};

    hahaha::Tensor<hahaha::common::u32> t2({3, 4});
    auto t3 = t1 * t2;
}

void tensor_basic_div_example() {
    hahaha::Tensor<hahaha::common::u32> t1 = {{1, 2}};

    hahaha::Tensor<hahaha::common::u32> t2({3, 4});
    auto t3 = t1 / t2;
}

void tensor_basic_matmul_example() {

    hahaha::Tensor<hahaha::common::u32> t1({{1, 2}, {3, 4}});
    hahaha::Tensor<hahaha::common::u32> t2({{1, 2}, {3, 4}});

    auto  t3 = t1.matmul(t2);
}



