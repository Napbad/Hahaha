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

#include "ml/util/dataset/DatasetType.h"

#include <gtest/gtest.h>

using namespace hahaha;

class DatasetTypeTest : public ::testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DatasetTypeTest, EnumValuesTest) {
    // Test that all expected enum values exist
    EXPECT_EQ(static_cast<int>(DatasetType::MNIST), 0);
    EXPECT_EQ(static_cast<int>(DatasetType::CIFAR10), 1);
    EXPECT_EQ(static_cast<int>(DatasetType::CIFAR100), 2);

    // Test that we can create variables of this type
    const auto mnistType    = DatasetType::MNIST;
    const auto cifar10Type  = DatasetType::CIFAR10;
    const auto cifar100Type = DatasetType::CIFAR100;

    // Test comparison
    EXPECT_NE(mnistType, cifar10Type);
    EXPECT_NE(cifar10Type, cifar100Type);
    EXPECT_NE(mnistType, cifar100Type);

    // Test assignment
    const DatasetType assignedType = mnistType;
    EXPECT_EQ(assignedType, mnistType);
}
