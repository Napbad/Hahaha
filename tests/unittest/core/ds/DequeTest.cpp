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

#include <gtest/gtest.h>
#include "core/ds/dqueue.h"
#include <string>

using namespace hahaha::core::ds;

class DequeTest : public ::testing::Test {
protected:
    Deque<int> intDeque;
    Deque<std::string> strDeque;
};

TEST_F(DequeTest, InitialState) {
    ASSERT_TRUE(intDeque.empty());
    ASSERT_EQ(intDeque.size(), 0);
}

TEST_F(DequeTest, PushBack) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    ASSERT_EQ(intDeque.size(), 3);
    ASSERT_FALSE(intDeque.empty());
    ASSERT_EQ(intDeque.front(), 1);
    ASSERT_EQ(intDeque.back(), 3);
}

TEST_F(DequeTest, PushFront) {
    intDeque.pushFront(1);
    intDeque.pushFront(2);
    intDeque.pushFront(3);
    
    ASSERT_EQ(intDeque.size(), 3);
    ASSERT_EQ(intDeque.front(), 3);
    ASSERT_EQ(intDeque.back(), 1);
}

TEST_F(DequeTest, MixedPushes) {
    intDeque.pushBack(1);
    intDeque.pushFront(2);
    intDeque.pushBack(3);
    intDeque.pushFront(4);
    // Order should be: 4, 2, 1, 3
    
    ASSERT_EQ(intDeque.size(), 4);
    ASSERT_EQ(intDeque.front(), 4);
    ASSERT_EQ(intDeque.back(), 3);
}

TEST_F(DequeTest, PopFront) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    intDeque.popFront();
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.front(), 2);
    
    intDeque.popFront();
    ASSERT_EQ(intDeque.size(), 1);
    ASSERT_EQ(intDeque.front(), 3);
    ASSERT_EQ(intDeque.back(), 3);
}

TEST_F(DequeTest, PopBack) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    intDeque.popBack();
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.back(), 2);
    
    intDeque.popBack();
    ASSERT_EQ(intDeque.size(), 1);
    ASSERT_EQ(intDeque.front(), 1);
    ASSERT_EQ(intDeque.back(), 1);
}

TEST_F(DequeTest, PopFrontEmpty) {
    ASSERT_THROW(intDeque.popFront(), std::out_of_range);
}

TEST_F(DequeTest, PopBackEmpty) {
    ASSERT_THROW(intDeque.popBack(), std::out_of_range);
}

TEST_F(DequeTest, FrontEmpty) {
    ASSERT_THROW(intDeque.front(), std::out_of_range);
}

TEST_F(DequeTest, BackEmpty) {
    ASSERT_THROW(intDeque.back(), std::out_of_range);
}

TEST_F(DequeTest, At) {
    intDeque.pushBack(10);
    intDeque.pushBack(20);
    intDeque.pushBack(30);
    intDeque.pushBack(40);
    
    ASSERT_EQ(intDeque.at(0), 10);
    ASSERT_EQ(intDeque.at(1), 20);
    ASSERT_EQ(intDeque.at(2), 30);
    ASSERT_EQ(intDeque.at(3), 40);
}

TEST_F(DequeTest, AtOutOfRange) {
    intDeque.pushBack(1);
    ASSERT_THROW(intDeque.at(1), std::out_of_range);
    ASSERT_THROW(intDeque.at(100), std::out_of_range);
}

TEST_F(DequeTest, OperatorBracket) {
    intDeque.pushBack(10);
    intDeque.pushBack(20);
    intDeque.pushBack(30);
    
    ASSERT_EQ(intDeque[0], 10);
    ASSERT_EQ(intDeque[1], 20);
    ASSERT_EQ(intDeque[2], 30);
}

TEST_F(DequeTest, ModifyElements) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    intDeque[1] = 99;
    ASSERT_EQ(intDeque[1], 99);
    
    intDeque.at(0) = 88;
    ASSERT_EQ(intDeque.at(0), 88);
}

TEST_F(DequeTest, Clear) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    ASSERT_EQ(intDeque.size(), 3);
    intDeque.clear();
    ASSERT_TRUE(intDeque.empty());
    ASSERT_EQ(intDeque.size(), 0);
}

TEST_F(DequeTest, InitializerList) {
    Deque<int> deque{1, 2, 3, 4, 5};
    
    ASSERT_EQ(deque.size(), 5);
    ASSERT_EQ(deque.front(), 1);
    ASSERT_EQ(deque.back(), 5);
    ASSERT_EQ(deque[2], 3);
}

TEST_F(DequeTest, CopyConstructor) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    Deque<int> copy(intDeque);
    ASSERT_EQ(copy.size(), 3);
    ASSERT_EQ(copy.front(), 1);
    ASSERT_EQ(copy.back(), 3);
    
    // Modify original, copy should be unchanged
    intDeque.popFront();
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(copy.size(), 3);
}

TEST_F(DequeTest, CopyAssignment) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    
    Deque<int> copy;
    copy = intDeque;
    
    ASSERT_EQ(copy.size(), 2);
    ASSERT_EQ(copy.front(), 1);
    ASSERT_EQ(copy.back(), 2);
}

TEST_F(DequeTest, MoveConstructor) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.pushBack(3);
    
    Deque<int> moved(std::move(intDeque));
    ASSERT_EQ(moved.size(), 3);
    ASSERT_EQ(moved.front(), 1);
    ASSERT_EQ(moved.back(), 3);
    
    // Original should be empty after move
    ASSERT_TRUE(intDeque.empty());
}

TEST_F(DequeTest, MoveAssignment) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    
    Deque<int> moved;
    moved = std::move(intDeque);
    
    ASSERT_EQ(moved.size(), 2);
    ASSERT_EQ(moved.front(), 1);
    ASSERT_TRUE(intDeque.empty());
}

TEST_F(DequeTest, StringDeque) {
    strDeque.pushBack("hello");
    strDeque.pushBack("world");
    strDeque.pushFront("hi");
    
    ASSERT_EQ(strDeque.size(), 3);
    ASSERT_EQ(strDeque.front(), "hi");
    ASSERT_EQ(strDeque.back(), "world");
    ASSERT_EQ(strDeque[1], "hello");
}

TEST_F(DequeTest, EmplaceFront) {
    intDeque.emplaceFront(1);
    intDeque.emplaceFront(2);
    
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.front(), 2);
    ASSERT_EQ(intDeque.back(), 1);
}

TEST_F(DequeTest, EmplaceBack) {
    intDeque.emplaceBack(1);
    intDeque.emplaceBack(2);
    
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.front(), 1);
    ASSERT_EQ(intDeque.back(), 2);
}

TEST_F(DequeTest, LargeDataSet) {
    for (int i = 0; i < 50; ++i) {
        intDeque.pushBack(i);
    }
    
    for (int i = 50; i < 100; ++i) {
        intDeque.pushFront(i);
    }
    
    ASSERT_EQ(intDeque.size(), 100);
    
    // Verify all elements
    for (int i = 0; i < 100; ++i) {
        ASSERT_NO_THROW(intDeque.at(i));
    }
}

TEST_F(DequeTest, AlternatingOperations) {
    intDeque.pushBack(1);
    intDeque.pushFront(2);
    intDeque.pushBack(3);
    intDeque.popFront();
    intDeque.pushFront(4);
    intDeque.popBack();
    
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.front(), 4);
    ASSERT_EQ(intDeque.back(), 1);
}

TEST_F(DequeTest, ClearAndReuse) {
    intDeque.pushBack(1);
    intDeque.pushBack(2);
    intDeque.clear();
    
    intDeque.pushBack(3);
    intDeque.pushBack(4);
    
    ASSERT_EQ(intDeque.size(), 2);
    ASSERT_EQ(intDeque.front(), 3);
    ASSERT_EQ(intDeque.back(), 4);
}

