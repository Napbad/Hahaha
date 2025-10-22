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

#include "core/ds/Map.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace hahaha::core::ds;

class MapTest : public ::testing::Test
{
  protected:
    Map<int, std::string> intStrMap;
    Map<std::string, int> strIntMap;
};

TEST_F(MapTest, InitialState)
{
    ASSERT_TRUE(intStrMap.empty());
    ASSERT_EQ(intStrMap.size(), 0);
}

TEST_F(MapTest, InsertAndAccess)
{
    auto [inserted1, val1] = intStrMap.insert(1, "one");
    ASSERT_TRUE(inserted1);
    ASSERT_EQ(*val1, "one");

    auto [inserted2, val2] = intStrMap.insert(2, "two");
    ASSERT_TRUE(inserted2);
    ASSERT_EQ(*val2, "two");

    ASSERT_EQ(intStrMap.size(), 2);
    ASSERT_FALSE(intStrMap.empty());
}

TEST_F(MapTest, OperatorBracket)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";
    intStrMap[3] = "three";

    ASSERT_EQ(intStrMap[1], "one");
    ASSERT_EQ(intStrMap[2], "two");
    ASSERT_EQ(intStrMap[3], "three");
    ASSERT_EQ(intStrMap.size(), 3);
}

TEST_F(MapTest, OperatorBracketCreatesDefault)
{
    std::string val = intStrMap[999];
    ASSERT_EQ(val, ""); // Default constructed string
    ASSERT_EQ(intStrMap.size(), 1);
}

TEST_F(MapTest, AtMethod)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";

    ASSERT_EQ(intStrMap.at(1), "one");
    ASSERT_EQ(intStrMap.at(2), "two");

    ASSERT_THROW(intStrMap.at(999), std::out_of_range);
}

TEST_F(MapTest, InsertDuplicate)
{
    auto [inserted1, val1] = intStrMap.insert(1, "one");
    ASSERT_TRUE(inserted1);

    auto [inserted2, val2] = intStrMap.insert(1, "uno");
    ASSERT_FALSE(inserted2); // Should fail, key exists
    ASSERT_EQ(*val2, "one"); // Original value preserved
    ASSERT_EQ(intStrMap.size(), 1);
}

TEST_F(MapTest, Erase)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";
    intStrMap[3] = "three";

    ASSERT_EQ(intStrMap.erase(2), 1);
    ASSERT_EQ(intStrMap.size(), 2);
    ASSERT_EQ(intStrMap.erase(2), 0); // Already erased

    ASSERT_EQ(intStrMap[1], "one");
    ASSERT_EQ(intStrMap[3], "three");
}

TEST_F(MapTest, Clear)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";
    intStrMap[3] = "three";

    ASSERT_EQ(intStrMap.size(), 3);
    intStrMap.clear();
    ASSERT_TRUE(intStrMap.empty());
    ASSERT_EQ(intStrMap.size(), 0);
}

TEST_F(MapTest, Contains)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";

    ASSERT_TRUE(intStrMap.contains(1));
    ASSERT_TRUE(intStrMap.contains(2));
    ASSERT_FALSE(intStrMap.contains(3));
}

TEST_F(MapTest, Count)
{
    intStrMap[1] = "one";

    ASSERT_EQ(intStrMap.count(1), 1);
    ASSERT_EQ(intStrMap.count(999), 0);
}

TEST_F(MapTest, Find)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";

    auto* val1 = intStrMap.find(1);
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(*val1, "one");

    auto* val3 = intStrMap.find(999);
    ASSERT_EQ(val3, nullptr);
}

TEST_F(MapTest, Keys)
{
    intStrMap[3] = "three";
    intStrMap[1] = "one";
    intStrMap[2] = "two";

    auto keys = intStrMap.keys();
    ASSERT_EQ(keys.size(), 3);
    // Keys should be in sorted order (in-order traversal of RBTree)
    ASSERT_EQ(keys[0], 1);
    ASSERT_EQ(keys[1], 2);
    ASSERT_EQ(keys[2], 3);
}

TEST_F(MapTest, Values)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";
    intStrMap[3] = "three";

    auto values = intStrMap.values();
    ASSERT_EQ(values.size(), 3);
    // Values in order of keys
    ASSERT_EQ(values[0], "one");
    ASSERT_EQ(values[1], "two");
    ASSERT_EQ(values[2], "three");
}

TEST_F(MapTest, Entries)
{
    intStrMap[1] = "one";
    intStrMap[2] = "two";

    auto entries = intStrMap.entries();
    ASSERT_EQ(entries.size(), 2);
    ASSERT_EQ(entries[0].first, 1);
    ASSERT_EQ(entries[0].second, "one");
    ASSERT_EQ(entries[1].first, 2);
    ASSERT_EQ(entries[1].second, "two");
}

TEST_F(MapTest, StringKeys)
{
    strIntMap["hello"] = 1;
    strIntMap["world"] = 2;
    strIntMap["test"] = 3;

    ASSERT_EQ(strIntMap["hello"], 1);
    ASSERT_EQ(strIntMap["world"], 2);
    ASSERT_EQ(strIntMap["test"], 3);
    ASSERT_EQ(strIntMap.size(), 3);
}

TEST_F(MapTest, ModifyValue)
{
    intStrMap[1] = "one";
    ASSERT_EQ(intStrMap[1], "one");

    intStrMap[1] = "uno";
    ASSERT_EQ(intStrMap[1], "uno");
    ASSERT_EQ(intStrMap.size(), 1);
}

TEST_F(MapTest, LargeDataSet)
{
    for (int i = 0; i < 100; ++i)
    {
        intStrMap[i] = "value" + std::to_string(i);
    }

    ASSERT_EQ(intStrMap.size(), 100);

    for (int i = 0; i < 100; ++i)
    {
        ASSERT_EQ(intStrMap[i], "value" + std::to_string(i));
    }
}

TEST_F(MapTest, Emplace)
{
    auto [inserted1, val1] = intStrMap.emplace(1, "one");
    ASSERT_TRUE(inserted1);
    ASSERT_EQ(*val1, "one");

    auto [inserted2, val2] = intStrMap.emplace(1, "uno");
    ASSERT_FALSE(inserted2);
    ASSERT_EQ(*val2, "one");
}

TEST_F(MapTest, InsertPair)
{
    auto [inserted, val] = intStrMap.insert(std::make_pair(1, "one"));
    ASSERT_TRUE(inserted);
    ASSERT_EQ(*val, "one");
    ASSERT_EQ(intStrMap.size(), 1);
}
