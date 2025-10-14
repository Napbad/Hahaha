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
#include "core/ds/set.h"
#include <string>

using namespace hahaha::core::ds;

class SetTest : public ::testing::Test {
protected:
    Set<int> intSet;
    Set<std::string> strSet;
};

TEST_F(SetTest, InitialState) {
    ASSERT_TRUE(intSet.empty());
    ASSERT_EQ(intSet.size(), 0);
}

TEST_F(SetTest, Insert) {
    ASSERT_TRUE(intSet.insert(1));
    ASSERT_TRUE(intSet.insert(2));
    ASSERT_TRUE(intSet.insert(3));
    
    ASSERT_EQ(intSet.size(), 3);
    ASSERT_FALSE(intSet.empty());
}

TEST_F(SetTest, InsertDuplicate) {
    ASSERT_TRUE(intSet.insert(1));
    ASSERT_FALSE(intSet.insert(1)); // Duplicate
    ASSERT_FALSE(intSet.insert(1)); // Duplicate again
    
    ASSERT_EQ(intSet.size(), 1);
}

TEST_F(SetTest, Contains) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    
    ASSERT_TRUE(intSet.contains(1));
    ASSERT_TRUE(intSet.contains(2));
    ASSERT_TRUE(intSet.contains(3));
    ASSERT_FALSE(intSet.contains(4));
}

TEST_F(SetTest, Count) {
    intSet.insert(1);
    intSet.insert(2);
    
    ASSERT_EQ(intSet.count(1), 1);
    ASSERT_EQ(intSet.count(2), 1);
    ASSERT_EQ(intSet.count(999), 0);
}

TEST_F(SetTest, Find) {
    intSet.insert(1);
    intSet.insert(2);
    
    auto* val1 = intSet.find(1);
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(*val1, 1);
    
    auto* val3 = intSet.find(999);
    ASSERT_EQ(val3, nullptr);
}

TEST_F(SetTest, Erase) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    
    ASSERT_EQ(intSet.erase(2), 1);
    ASSERT_EQ(intSet.size(), 2);
    ASSERT_FALSE(intSet.contains(2));
    
    ASSERT_EQ(intSet.erase(2), 0); // Already erased
    ASSERT_EQ(intSet.erase(999), 0); // Never existed
}

TEST_F(SetTest, Clear) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    
    ASSERT_EQ(intSet.size(), 3);
    intSet.clear();
    ASSERT_TRUE(intSet.empty());
    ASSERT_EQ(intSet.size(), 0);
}

TEST_F(SetTest, MinMax) {
    intSet.insert(5);
    intSet.insert(1);
    intSet.insert(9);
    intSet.insert(3);
    
    auto* minVal = intSet.min();
    auto* maxVal = intSet.max();
    
    ASSERT_NE(minVal, nullptr);
    ASSERT_NE(maxVal, nullptr);
    ASSERT_EQ(*minVal, 1);
    ASSERT_EQ(*maxVal, 9);
}

TEST_F(SetTest, MinMaxEmpty) {
    ASSERT_EQ(intSet.min(), nullptr);
    ASSERT_EQ(intSet.max(), nullptr);
}

TEST_F(SetTest, ToVector) {
    intSet.insert(3);
    intSet.insert(1);
    intSet.insert(2);
    
    auto vec = intSet.toVector();
    ASSERT_EQ(vec.size(), 3);
    // Should be sorted (in-order traversal)
    ASSERT_EQ(vec[0], 1);
    ASSERT_EQ(vec[1], 2);
    ASSERT_EQ(vec[2], 3);
}

TEST_F(SetTest, UnionWith) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    
    Set<int> other;
    other.insert(3);
    other.insert(4);
    other.insert(5);
    
    auto result = intSet.unionWith(other);
    ASSERT_EQ(result.size(), 5);
    ASSERT_TRUE(result.contains(1));
    ASSERT_TRUE(result.contains(2));
    ASSERT_TRUE(result.contains(3));
    ASSERT_TRUE(result.contains(4));
    ASSERT_TRUE(result.contains(5));
}

TEST_F(SetTest, IntersectWith) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    intSet.insert(4);
    
    Set<int> other;
    other.insert(3);
    other.insert(4);
    other.insert(5);
    other.insert(6);
    
    auto result = intSet.intersectWith(other);
    ASSERT_EQ(result.size(), 2);
    ASSERT_TRUE(result.contains(3));
    ASSERT_TRUE(result.contains(4));
    ASSERT_FALSE(result.contains(1));
    ASSERT_FALSE(result.contains(5));
}

TEST_F(SetTest, Difference) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    intSet.insert(4);
    
    Set<int> other;
    other.insert(3);
    other.insert(4);
    other.insert(5);
    
    auto result = intSet.difference(other);
    ASSERT_EQ(result.size(), 2);
    ASSERT_TRUE(result.contains(1));
    ASSERT_TRUE(result.contains(2));
    ASSERT_FALSE(result.contains(3));
    ASSERT_FALSE(result.contains(4));
}

TEST_F(SetTest, IsSubsetOf) {
    intSet.insert(1);
    intSet.insert(2);
    
    Set<int> superset;
    superset.insert(1);
    superset.insert(2);
    superset.insert(3);
    superset.insert(4);
    
    ASSERT_TRUE(intSet.isSubsetOf(superset));
    ASSERT_FALSE(superset.isSubsetOf(intSet));
}

TEST_F(SetTest, IsSupersetOf) {
    intSet.insert(1);
    intSet.insert(2);
    intSet.insert(3);
    intSet.insert(4);
    
    Set<int> subset;
    subset.insert(1);
    subset.insert(2);
    
    ASSERT_TRUE(intSet.isSupersetOf(subset));
    ASSERT_FALSE(subset.isSupersetOf(intSet));
}

TEST_F(SetTest, StringSet) {
    strSet.insert("hello");
    strSet.insert("world");
    strSet.insert("test");
    
    ASSERT_EQ(strSet.size(), 3);
    ASSERT_TRUE(strSet.contains("hello"));
    ASSERT_TRUE(strSet.contains("world"));
    ASSERT_TRUE(strSet.contains("test"));
    ASSERT_FALSE(strSet.contains("notfound"));
}

TEST_F(SetTest, Emplace) {
    ASSERT_TRUE(intSet.emplace(1));
    ASSERT_TRUE(intSet.emplace(2));
    ASSERT_FALSE(intSet.emplace(1)); // Duplicate
    
    ASSERT_EQ(intSet.size(), 2);
}

TEST_F(SetTest, LargeDataSet) {
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(intSet.insert(i));
    }
    
    ASSERT_EQ(intSet.size(), 100);
    
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(intSet.contains(i));
    }
}

TEST_F(SetTest, EmptySetOperations) {
    Set<int> empty1;
    Set<int> empty2;
    
    auto unionResult = empty1.unionWith(empty2);
    ASSERT_TRUE(unionResult.empty());
    
    auto intersectResult = empty1.intersectWith(empty2);
    ASSERT_TRUE(intersectResult.empty());
    
    auto diffResult = empty1.difference(empty2);
    ASSERT_TRUE(diffResult.empty());
}

