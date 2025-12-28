// Copyright (c) 2025 Napbad
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
// Email:  napbad.sen@gmail.com 
// GitHub:  https://github.com/Napbad 
// 
// Created: 2025-12-23 06:51:07 by Napbad
// 

#include "math/ds/TensorShape.h"
#include <gtest/gtest.h>

class TensorShapeTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Objects declared here can be used by all tests in the test suite for Foo.
};

using namespace hahaha::math;
using namespace hahaha::common;

TEST_F(TensorShapeTest, InitWithInitializerList) {
    auto ts1 = TensorShape({1, 2, 3});
    ASSERT_EQ(ts1.dims().size(), 3);
}