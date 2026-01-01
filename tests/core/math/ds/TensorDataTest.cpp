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

#include "math/ds/TensorData.h"

#include <gtest/gtest.h>
class TensorDataTest : public ::testing::Test {
  protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

using hahaha::math::TensorData;

TEST_F(TensorDataTest, InitWithInitializerList) {
    TensorData<int> singleValueTensor(hahaha::math::NestedData<int>{1});
    EXPECT_EQ(singleValueTensor.getShape().getDims().size(), 1);
    EXPECT_EQ(singleValueTensor.getShape().getDims()[0], 1);
    EXPECT_EQ(singleValueTensor.getData()[0], 1);

    TensorData<int> twoElementTensor(hahaha::math::NestedData<int>{{1}, {2}});
    EXPECT_EQ(twoElementTensor.getShape().getDims().size(), 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[0], 2);
    EXPECT_EQ(twoElementTensor.getShape().getDims()[1], 1);
    EXPECT_EQ(twoElementTensor.getData()[0], 1);
    EXPECT_EQ(twoElementTensor.getData()[1], 2);
}

TEST_F(TensorDataTest, DefaultConstructor) {
    TensorData<float> defaultConstructedTensor;
    EXPECT_EQ(defaultConstructedTensor.getShape().getDims().size(), 0);
    EXPECT_EQ(defaultConstructedTensor.getData().get(), nullptr);
}

TEST_F(TensorDataTest, ShapeValueConstructor) {
    hahaha::math::TensorShape shape({2, 3});
    TensorData<int> tensor_data(shape, 7);
    EXPECT_EQ(tensor_data.getShape().getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(tensor_data.getData()[i], 7);
    }
    EXPECT_EQ(tensor_data.getStride().getSize(), 2);
    EXPECT_EQ(tensor_data.getStride()[0], 3);
    EXPECT_EQ(tensor_data.getStride()[1], 1);
}

TEST_F(TensorDataTest, OneDimensionalTensor) {
    TensorData<int> tensor_1d(hahaha::math::NestedData<int>{1, 2, 3, 4, 5});
    EXPECT_EQ(tensor_1d.getShape().getTotalSize(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(tensor_1d.getData()[i], i + 1);
    }
}

TEST_F(TensorDataTest, CopyConstructor) {
    TensorData<int> original(hahaha::math::NestedData<int>{{1, 2}, {3, 4}});
    TensorData<int> copied(original);

    EXPECT_EQ(copied.getShape(), original.getShape());
    EXPECT_EQ(copied.getStride().getSize(), original.getStride().getSize());
    for (size_t i = 0; i < original.getShape().getTotalSize(); ++i) {
        EXPECT_EQ(copied.getData()[i], original.getData()[i]);
    }

    // Verify deep copy
    copied.getData()[0] = 100;
    EXPECT_EQ(original.getData()[0], 1);
    EXPECT_EQ(copied.getData()[0], 100);
}

TEST_F(TensorDataTest, MoveConstructor) {
    TensorData<int> original(hahaha::math::NestedData<int>{1, 2, 3});
    void* originalPtr = original.getData().get();

    TensorData<int> moved(std::move(original));

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
    EXPECT_EQ(original.getShape().getDims().size(), 0);
}

TEST_F(TensorDataTest, MoveAssignment) {
    TensorData<int> original(hahaha::math::NestedData<int>{1, 2, 3});
    void* originalPtr = original.getData().get();
    TensorData<int> moved;

    moved = std::move(original);

    EXPECT_EQ(moved.getShape().getTotalSize(), 3);
    EXPECT_EQ(moved.getData().get(), originalPtr);
    EXPECT_EQ(original.getData().get(), nullptr);
}

TEST_F(TensorDataTest, SettersAndGetters) {
    TensorData<int> td;
    auto data = std::make_unique<int[]>(4);
    data[0] = 10;

    td.setData(std::move(data));
    td.setShape(hahaha::math::TensorShape({2, 2}));
    td.setStride(hahaha::math::TensorStride(hahaha::math::TensorShape({2, 2})));

    EXPECT_EQ(td.getData()[0], 10);
    EXPECT_EQ(td.getShape().getTotalSize(), 4);
    EXPECT_EQ(td.getStride()[0], 2);
}
