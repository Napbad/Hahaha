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
// Created by napbad on 10/14/25.
//
#include "core/ds/Stack.h"

#include <gtest/gtest.h>

using namespace hahaha::core::ds;

// Test fixture for Stack
class StackTest : public testing::Test
{
  protected:
    Stack<int> s1;
    Stack<int> s2;

    void SetUp() override
    {
        s2.push(1);
        s2.push(2);
        s2.push(3);
    }
};

TEST_F(StackTest, DefaultConstructor)
{
    EXPECT_TRUE(s1.empty());
    EXPECT_EQ(s1.size(), 0);
}

TEST_F(StackTest, InitializerListConstructor)
{
    Stack s({10, 20, 30});
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(s.size(), 3);
    EXPECT_EQ(s.top(), 30);
}

TEST_F(StackTest, Push)
{
    s1.push(100);
    EXPECT_FALSE(s1.empty());
    EXPECT_EQ(s1.size(), 1);
    EXPECT_EQ(s1.top(), 100);

    s1.push(200);
    EXPECT_EQ(s1.size(), 2);
    EXPECT_EQ(s1.top(), 200);
}

TEST_F(StackTest, Pop)
{
    EXPECT_EQ(s2.size(), 3);
    EXPECT_EQ(s2.top(), 3);

    s2.pop();
    EXPECT_EQ(s2.size(), 2);
    EXPECT_EQ(s2.top(), 2);

    s2.pop();
    EXPECT_EQ(s2.size(), 1);
    EXPECT_EQ(s2.top(), 1);

    s2.pop();
    EXPECT_TRUE(s2.empty());
}

TEST_F(StackTest, LIFOOrder)
{
    Stack<std::string> s;
    s.push("First");
    s.push("Second");
    s.push("Third");

    EXPECT_EQ(s.top(), "Third");
    s.pop();
    EXPECT_EQ(s.top(), "Second");
    s.pop();
    EXPECT_EQ(s.top(), "First");
}

TEST_F(StackTest, TopOnEmpty)
{
    EXPECT_THROW(s1.top(), std::runtime_error);
}

TEST_F(StackTest, PopOnEmpty)
{
    EXPECT_THROW(s1.pop(), std::runtime_error);
}

TEST_F(StackTest, ConstCorrectness)
{
    const Stack const_stack({5, 10});

    EXPECT_EQ(const_stack.size(), 2);
    EXPECT_FALSE(const_stack.empty());
    EXPECT_EQ(const_stack.top(), 10);
}

TEST_F(StackTest, ModifyTop)
{
    s2.top() = 33;
    EXPECT_EQ(s2.top(), 33);
}

TEST_F(StackTest, SizeAndCapacity)
{
    EXPECT_EQ(s2.size(), 3);
    EXPECT_GE(s2.capacity(), 3);
    s2.push(4);
    EXPECT_EQ(s2.size(), 4);
    EXPECT_GE(s2.capacity(), 4);
}