// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "math/ds/TensorShape.h"

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>

class TensorShapeTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }
    void TearDown() override {
    }
};

using hahaha::common::u32;
using hahaha::math::TensorShape;

TEST_F(TensorShapeTest, InitWithInitializerList) {
    auto ts1 = TensorShape({1, 2, 3});
    ASSERT_EQ(ts1.getDims().size(), 3);
    ASSERT_EQ(ts1.getDims().at(0), 1);
    ASSERT_EQ(ts1.getDims().at(1), 2);
    ASSERT_EQ(ts1.getDims().at(2), 3);
}

TEST_F(TensorShapeTest, InitWithNoParam) {
    auto ts1 = TensorShape();
    ASSERT_EQ(ts1.getDims().size(), 0);
}

TEST_F(TensorShapeTest, InitWithStdVector) {
    std::vector<size_t> dims = {4, 5, 6, 7};
    auto ts1 = TensorShape(dims);
    ASSERT_EQ(ts1.getDims().size(), 4);
    ASSERT_EQ(ts1.getDims().at(0), 4);
    ASSERT_EQ(ts1.getDims().at(1), 5);
    ASSERT_EQ(ts1.getDims().at(2), 6);
    ASSERT_EQ(ts1.getDims().at(3), 7);
}

TEST_F(TensorShapeTest, InitWithEmptyStdVector_IsScalarShape) {
    std::vector<size_t> dims;
    auto ts = TensorShape(dims);
    ASSERT_EQ(ts.getDims().size(), 0);
    ASSERT_EQ(ts.getTotalSize(), 1);
    ASSERT_EQ(ts.toString(), "()");
}

TEST_F(TensorShapeTest, MoveConstructor) {
    TensorShape ts1({8, 9, 10});
    TensorShape ts2(std::move(ts1));
    ASSERT_EQ(ts1.getDims().size(), 0);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 8);
    ASSERT_EQ(ts2.getDims().at(1), 9);
    ASSERT_EQ(ts2.getDims().at(2), 10);
}

TEST_F(TensorShapeTest, CopyConstructor) {
    TensorShape ts1({11, 12, 13});
    TensorShape ts2(ts1);
    ASSERT_EQ(ts1.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 11);
    ASSERT_EQ(ts2.getDims().at(1), 12);
    ASSERT_EQ(ts2.getDims().at(2), 13);
}

TEST_F(TensorShapeTest, AssignWithLValue) {
    TensorShape ts1({14, 15});
    TensorShape ts2;
    ASSERT_EQ(ts2.getDims().size(), 0);
    ts2 = ts1;
    ASSERT_EQ(ts1.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().at(0), 14);
    ASSERT_EQ(ts2.getDims().at(1), 15);
}

TEST_F(TensorShapeTest, AssignWithRValue) {
    TensorShape ts1({16, 17, 18});
    TensorShape ts2;
    ASSERT_EQ(ts2.getDims().size(), 0);
    ts2 = std::move(ts1);
    ASSERT_EQ(ts1.getDims().size(), 0);
    ASSERT_EQ(ts2.getDims().size(), 3);
    ASSERT_EQ(ts2.getDims().at(0), 16);
    ASSERT_EQ(ts2.getDims().at(1), 17);
    ASSERT_EQ(ts2.getDims().at(2), 18);
    ts2 = TensorShape({19, 20});
    ASSERT_EQ(ts2.getDims().size(), 2);
    ASSERT_EQ(ts2.getDims().at(0), 19);
    ASSERT_EQ(ts2.getDims().at(1), 20);
}

TEST_F(TensorShapeTest, ComputeSize) {
    TensorShape ts1({2, 3, 4});
    ASSERT_EQ(ts1.getTotalSize(), 24);

    TensorShape ts2({5, 6});
    ASSERT_EQ(ts2.getTotalSize(), 30);

    TensorShape ts3;
    ASSERT_EQ(ts3.getTotalSize(), 1);

    TensorShape ts4({1024, 1024, 1024, 8});
    ASSERT_EQ(ts4.getTotalSize(), 8589934592ULL);
}

TEST_F(TensorShapeTest, DimsAccess) {
    TensorShape ts({2, 3});
    const auto& dims = ts.getDims();
    ASSERT_EQ(dims.size(), 2);
    ASSERT_EQ(dims[0], 2);
    ASSERT_EQ(dims[1], 3);

    ASSERT_EQ(&ts.getDims(), &ts.getDims());
}

TEST_F(TensorShapeTest, ToString) {
    TensorShape ts1({1, 2, 3});
    ASSERT_EQ(ts1.toString(), "(1, 2, 3)");

    TensorShape ts2({4, 5});
    ASSERT_EQ(ts2.toString(), "(4, 5)");

    TensorShape ts3;
    ASSERT_EQ(ts3.toString(), "()");
}

TEST_F(TensorShapeTest, Reverse) {
    TensorShape ts1({1, 2, 3});
    ts1.reverse();
    ASSERT_EQ(ts1.toString(), "(3, 2, 1)");

    TensorShape ts2({4, 5});
    ts2.reverse();
    ASSERT_EQ(ts2.toString(), "(5, 4)");
}

TEST_F(TensorShapeTest, OperatorEqual) {
    ASSERT_TRUE(TensorShape({1, 2, 3}) == TensorShape({1, 2, 3}));
}

TEST_F(TensorShapeTest, OperatorNotEqual) {
    ASSERT_TRUE(TensorShape({1, 2, 3}) != TensorShape({1, 2, 4}));
}

TEST_F(TensorShapeTest, BroadcastShape_SameShape_ReturnsSame) {
    auto res = TensorShape::broadcastShape(TensorShape({2, 3, 4}),
                                           TensorShape({2, 3, 4}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3, 4}));
}

TEST_F(TensorShapeTest, BroadcastShape_ScalarWithTensor_ReturnsTensorShape) {
    // scalar is rank-0 (dims == {})
    auto res = TensorShape::broadcastShape(TensorShape({}),
                                           TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, BroadcastShape_PrefixDims_ReturnsTargetShape) {
    // (3) with (2,3) -> (2,3)
    auto res =
        TensorShape::broadcastShape(TensorShape({3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, BroadcastShape_DimOneBroadcasts) {
    // (1,3) with (2,3) -> (2,3)
    auto res =
        TensorShape::broadcastShape(TensorShape({1, 3}), TensorShape({2, 3}));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(TensorShape(*res), TensorShape({2, 3}));
}

TEST_F(TensorShapeTest, BroadcastShape_Incompatible_ReturnsNullopt) {
    auto res =
        TensorShape::broadcastShape(TensorShape({2, 3}), TensorShape({4, 3}));
    ASSERT_FALSE(res.has_value());
}
