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
// Created: 2025-12-23 06:51:07 by Napbad
// 

#include <iostream>

#include <gtest/gtest.h>

#include "math/ds/TensorShape.h"

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

using hahaha::math::TensorShape;
using hahaha::common::u32;

TEST_F(TensorShapeTest, InitWithInitializerList) {
    auto ts1 = TensorShape({1, 2, 3});
    ASSERT_EQ(ts1.dims().size(), 3);
    ASSERT_EQ(ts1.dims().at(0), 1);
    ASSERT_EQ(ts1.dims().at(1), 2);
    ASSERT_EQ(ts1.dims().at(2), 3);
}

TEST_F(TensorShapeTest, InitWithNoParam) {
    auto ts1 = TensorShape();
    ASSERT_EQ(ts1.dims().size(), 0);
}

TEST_F(TensorShapeTest, InitWithStdVector) {
    std::vector<u32> dims = {4, 5, 6, 7};
    std::cout << "Vector size: " << dims.size() << std::endl;
    auto ts1 = TensorShape(dims);
    ASSERT_EQ(ts1.dims().size(), 4);
    ASSERT_EQ(ts1.dims().at(0), 4);
    ASSERT_EQ(ts1.dims().at(1), 5);
    ASSERT_EQ(ts1.dims().at(2), 6);
    ASSERT_EQ(ts1.dims().at(3), 7);
}

TEST_F(TensorShapeTest, MoveConstructor) {
    TensorShape ts1({8, 9, 10});
    TensorShape ts2(std::move(ts1));
    ASSERT_EQ(ts1.dims().size(), 0);
    ASSERT_EQ(ts2.dims().size(), 3);
    ASSERT_EQ(ts2.dims().at(0), 8);
    ASSERT_EQ(ts2.dims().at(1), 9);
    ASSERT_EQ(ts2.dims().at(2), 10);
}

TEST_F(TensorShapeTest, CopyConstructor) {
    TensorShape ts1({11, 12, 13});
    TensorShape ts2(ts1);
    ASSERT_EQ(ts1.dims().size(), 3);
    ASSERT_EQ(ts2.dims().size(), 3);
    ASSERT_EQ(ts2.dims().at(0), 11);
    ASSERT_EQ(ts2.dims().at(1), 12);
    ASSERT_EQ(ts2.dims().at(2), 13);
}

TEST_F(TensorShapeTest, AssignWithLValue) {
    TensorShape ts1({14, 15});
    TensorShape ts2;
    ASSERT_EQ(ts2.dims().size(), 0);
    ts2 = ts1;
    ASSERT_EQ(ts1.dims().size(), 2);
    ASSERT_EQ(ts2.dims().size(), 2);
    ASSERT_EQ(ts2.dims().at(0), 14);
    ASSERT_EQ(ts2.dims().at(1), 15);
}

TEST_F(TensorShapeTest, AssignWithRValue) {
    TensorShape ts1({16, 17, 18});
    TensorShape ts2;
    ASSERT_EQ(ts2.dims().size(), 0);
    ts2 = std::move(ts1);
    ASSERT_EQ(ts1.dims().size(), 0);
    ASSERT_EQ(ts2.dims().size(), 3);
    ASSERT_EQ(ts2.dims().at(0), 16);
    ASSERT_EQ(ts2.dims().at(1), 17);
    ASSERT_EQ(ts2.dims().at(2), 18);
    ts2 = TensorShape({19, 20});
    ASSERT_EQ(ts2.dims().size(), 2);
    ASSERT_EQ(ts2.dims().at(0), 19);
    ASSERT_EQ(ts2.dims().at(1), 20);
}

TEST_F(TensorShapeTest, ComputeSize) {
    TensorShape ts1({2, 3, 4});
    ASSERT_EQ(ts1.totalSize(), 24);

    TensorShape ts2({5, 6});
    ASSERT_EQ(ts2.totalSize(), 30);

    TensorShape ts3;
    ASSERT_EQ(ts3.totalSize(), 1);
}

TEST_F(TensorShapeTest, ToString) {
    TensorShape ts1({1, 2, 3});
    ASSERT_EQ(ts1.toString(), "(1, 2, 3)");

    TensorShape ts2({4, 5});
    ASSERT_EQ(ts2.toString(), "(4, 5)");

    TensorShape ts3;
    ASSERT_EQ(ts3.toString(), "()");
}
