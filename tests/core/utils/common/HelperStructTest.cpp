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
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
// 

#include <gtest/gtest.h>
#include "utils/common/HelperStruct.h"
#include <initializer_list>

using namespace hahaha::utils;

TEST(HelperStructTest, IsInitList) {
    EXPECT_TRUE(isInitList<std::initializer_list<int>>::value);
    EXPECT_FALSE(isInitList<int>::value);
    EXPECT_FALSE(isInitList<std::vector<int>>::value);
}

TEST(HelperStructTest, IsNestedInitList) {
    EXPECT_TRUE(isNestedInitList<std::initializer_list<std::initializer_list<int>>>::value);
    EXPECT_FALSE(isNestedInitList<std::initializer_list<int>>::value);
}

TEST(HelperStructTest, IsLegalDataType) {
    EXPECT_TRUE(isLegalDataType<uint8_t>::value);
    EXPECT_TRUE(isLegalDataType<int32_t>::value);
    EXPECT_TRUE(isLegalDataType<float>::value);
    EXPECT_TRUE(isLegalDataType<double>::value);
    EXPECT_FALSE(isLegalDataType<std::string>::value);
}

