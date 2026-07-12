//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#include <assert.h>
#include <iostream>

#include "defines.h"
#include "ml/Tensor.h"

bool PrintInfo = true;

void test_init() {
    h3::core::ml::Tensor t1({1, 2}, h3::core::DataType::Float32);


    auto tmp1 = t1[0];
    auto tmp2 = tmp1[0];
    tmp2 = 1;
    if (PrintInfo) {
        std::cout << t1 << std::endl;
        std::cout << tmp2.item().value() << std::endl;
    }

    h3::core::ml::Tensor t2(std::vector<h3::core::SizeT>{}, h3::core::DataType::Float32);
    t2 = 10;

    assert(*tmp2.item().value().as<float>() == 1.0);
    assert(*t2.item().value().as<float>() == 10.0);
    if (PrintInfo) {
        std::cout << t2 << std::endl;
        std::cout << t2.item().value() << std::endl;
    }


}

void test_assign() {

    h3::core::ml::Tensor t1({2, 2}, h3::core::DataType::Float32);
    auto tmp1 = t1[0];
    auto tmp2 = t1[1];

    tmp1[0] = 10;

    assert(*tmp1[0].item().value().as<float>() == 10.0);

    assert(*t1[0][0].item().value().as<float>() == 10.0);
    tmp2[0] = 1;
    tmp1 = tmp2;

    assert(*tmp1[0].item().value().as<float>() == 1.0);
    assert(*tmp2[0].item().value().as<float>() == 1.0);

    assert(*t1[0][0].item().value().as<float>() == 10.0);
    assert(*t1[1][0].item().value().as<float>() == 1.0);

    if (PrintInfo) {
        std::cout << t1 << std::endl;
        std::cout << "tmp1[0] = " << tmp1[0].item().value() << std::endl;
        std::cout << "tmp1[1] = " << tmp1[1].item().value() << std::endl;
        std::cout << "tmp2[0] = " << tmp2[0].item().value() << std::endl;
        std::cout << "tmp2[1] = " << tmp2[1].item().value() << std::endl;

        std::cout << "t1[0][0] = " << t1[0][0].item().value() << std::endl;
        std::cout << "t1[0][1] = " << t1[0][1].item().value() << std::endl;
        std::cout << "t1[1][0] = " << t1[1][0].item().value() << std::endl;
        std::cout << "t1[1][1] = " << t1[1][1].item().value() << std::endl;
    }
}

int main() {
    std::cout << "Test Init" << std::endl;
    test_init();
    std::cout << "Test Assign" << std::endl;
    test_assign();
    return 0;
}
