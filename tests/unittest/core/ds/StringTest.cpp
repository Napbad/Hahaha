#include <Error.h>
#include <ds/String.h>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using namespace hahaha::core;
using namespace hahaha::core::ds;
using hahaha::sizeT;

// Test fixture for String
class StringTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // core setup for tests
    }

    void TearDown() override
    {
        // core teardown for tests
    }
};

TEST_F(StringTest, DefaultConstructor)
{
    String s;
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);
    ASSERT_STREQ(s.cStr(), "");
}

TEST_F(StringTest, CStringConstructor)
{
    String s("hello");
    ASSERT_FALSE(s.empty());
    ASSERT_EQ(s.size(), 5);
    ASSERT_STREQ(s.cStr(), "hello");
}

TEST_F(StringTest, CopyConstructor)
{
    String original("world");
    String copied = original;
    ASSERT_EQ(copied.size(), 5);
    ASSERT_STREQ(copied.cStr(), "world");
    // Ensure deep copy
    copied[0] = 'W';
    ASSERT_STREQ(original.cStr(), "world");
    ASSERT_STREQ(copied.cStr(), "World");
}

TEST_F(StringTest, MoveConstructor)
{
    String original("moveme");
    char* original_data = original.begin();
    String moved = std::move(original);
    ASSERT_EQ(moved.size(), 6);
    ASSERT_STREQ(moved.cStr(), "moveme");
    ASSERT_TRUE(original.empty());
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(moved.begin(), original_data);
}

TEST_F(StringTest, IteratorConstructor)
{
    char arr[] = "test";
    String s(arr, arr + 4);
    ASSERT_EQ(s.size(), 4);
    ASSERT_STREQ(s.cStr(), "test");
}

TEST_F(StringTest, StdStringConstructor)
{
    std::string std_s = "standard";
    String s(std_s);
    ASSERT_EQ(s.size(), 8);
    ASSERT_STREQ(s.cStr(), "standard");
}

TEST_F(StringTest, StringStreamConstructor)
{
    std::stringstream ss;
    ss << "stream";
    String s(ss);
    ASSERT_EQ(s.size(), 6);
    ASSERT_STREQ(s.cStr(), "stream");
}

TEST_F(StringTest, OStringStreamConstructor)
{
    std::ostringstream oss;
    oss << "ostringstream";
    String s(oss);
    ASSERT_EQ(s.size(), 13);
    ASSERT_STREQ(s.cStr(), "ostringstream");
}

TEST_F(StringTest, Destructor)
{
    // Implicitly tested by no crashes on scope exit
    String* s = new String("heap");
    delete s;
}

TEST_F(StringTest, CopyAssignment)
{
    String original("source");
    String assigned;
    assigned = original;
    ASSERT_EQ(assigned.size(), 6);
    ASSERT_STREQ(assigned.cStr(), "source");
    assigned[0] = 'S';
    ASSERT_STREQ(original.cStr(), "source"); // Deep copy
}

TEST_F(StringTest, MoveAssignment)
{
    String original("moving");
    char* original_data = original.begin();
    String assigned;
    assigned = std::move(original);
    ASSERT_EQ(assigned.size(), 6);
    ASSERT_STREQ(assigned.cStr(), "moving");
    ASSERT_TRUE(original.empty());
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(assigned.begin(), original_data);
}

TEST_F(StringTest, SizeLengthCapacityEmpty)
{
    String s("12345");
    ASSERT_EQ(s.size(), 5);
    ASSERT_EQ(s.length(), 5);
    ASSERT_GE(s.capacity(), 5);
    ASSERT_FALSE(s.empty());

    String empty_s;
    ASSERT_TRUE(empty_s.empty());
    ASSERT_EQ(empty_s.size(), 0);
}

TEST_F(StringTest, AtOperator)
{
    String s("abc");
    ASSERT_EQ(s.at(0).unwrap(), 'a');
    ASSERT_EQ(s.at(2).unwrap(), 'c');
    ASSERT_TRUE(s.at(3).isErr());
    ASSERT_EQ(s[0], 'a');
    ASSERT_EQ(s[2], 'c');
}

TEST_F(StringTest, BeginEnd)
{
    String s("abc");
    ASSERT_EQ(*s.begin(), 'a');
    ASSERT_EQ(*(s.end() - 1), 'c');
    ASSERT_EQ(s.end() - s.begin(), s.size());
}

TEST_F(StringTest, Clear)
{
    String s("not empty");
    s.clear();
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);
    ASSERT_STREQ(s.cStr(), "");
}

TEST_F(StringTest, Resize)
{
    String s("abc");
    s.resize(5, 'x');
    ASSERT_EQ(s.size(), 5);
    ASSERT_STREQ(s.cStr(), "abcxx");

    s.resize(2);
    ASSERT_EQ(s.size(), 2);
    ASSERT_STREQ(s.cStr(), "ab");

    String s2;
    s2.resize(3, 'z');
    ASSERT_EQ(s2.size(), 3);
    ASSERT_STREQ(s2.cStr(), "zzz");
}

TEST_F(StringTest, CStrData)
{
    String s("hello");
    ASSERT_STREQ(s.cStr(), "hello");
    ASSERT_STREQ(s.data(), "hello");
}

TEST_F(StringTest, PlusEqualsOperator)
{
    String s("hello");
    s += " world";
    ASSERT_STREQ(s.cStr(), "hello world");
    ASSERT_EQ(s.size(), 11);

    s += '!';
    ASSERT_STREQ(s.cStr(), "hello world!");
    ASSERT_EQ(s.size(), 12);

    String s2("foo");
    String s3("bar");
    s2 += s3;
    ASSERT_STREQ(s2.cStr(), "foobar");
    ASSERT_EQ(s2.size(), 6);
}

TEST_F(StringTest, EqualityOperators)
{
    String s1("test");
    String s2("test");
    String s3("Test");
    String s4("testt");

    ASSERT_TRUE(s1 == s2);
    ASSERT_FALSE(s1 == s3);
    ASSERT_FALSE(s1 == s4);
    ASSERT_TRUE(s1 != s3);
}

TEST_F(StringTest, ComparisonOperators)
{
    String s1("apple");
    String s2("banana");
    String s3("apple");

    ASSERT_TRUE(s1 < s2);
    ASSERT_FALSE(s2 < s1);
    ASSERT_TRUE(s1 <= s3);
    ASSERT_TRUE(s2 > s1);
    ASSERT_FALSE(s1 > s2);
    ASSERT_TRUE(s2 >= s1);
}

TEST_F(StringTest, PushBack)
{
    String s("a");
    s.push_back('b');
    ASSERT_STREQ(s.cStr(), "ab");
    ASSERT_EQ(s.size(), 2);
}

TEST_F(StringTest, InsertChar)
{
    String s("ace");
    s.insert(1, 'b');
    ASSERT_STREQ(s.cStr(), "abce");
    ASSERT_EQ(s.size(), 4);

    s.insert(0, 'z');
    ASSERT_STREQ(s.cStr(), "zabce");

    s.insert(s.size(), 'w');
    ASSERT_STREQ(s.cStr(), "zabcew");

    String s2("hello");
    s2.insert(10, 'x'); // Insert at end if pos > size
    ASSERT_STREQ(s2.cStr(), "hellox");

    String empty;
    empty.insert(0, 'a');
    ASSERT_STREQ(empty.cStr(), "a");
}

TEST_F(StringTest, InsertString)
{
    String s("ac");
    String to_insert("bb");
    s.insert(1, to_insert);
    ASSERT_STREQ(s.cStr(), "abbc");
    ASSERT_EQ(s.size(), 4);

    s.insert(0, String("xx"));
    ASSERT_STREQ(s.cStr(), "xxabbc");

    s.insert(s.size(), String("yy"));
    ASSERT_STREQ(s.cStr(), "xxabbcyy");

    String s2("test");
    s2.insert(10, String("end")); // Insert at end if pos > size
    ASSERT_STREQ(s2.cStr(), "testend");

    String s3("initial");
    s3.insert(3, String("")); // Inserting empty string should not change
    ASSERT_STREQ(s3.cStr(), "initial");
}

TEST_F(StringTest, Erase)
{
    String s("abcdefg");
    s.erase(2, 2); // Remove cd
    ASSERT_STREQ(s.cStr(), "abefg");
    ASSERT_EQ(s.size(), 5);

    s.erase(0, 1); // Remove a
    ASSERT_STREQ(s.cStr(), "befg");
    ASSERT_EQ(s.size(), 4);

    s.erase(3); // Remove g (default count 1)
    ASSERT_STREQ(s.cStr(), "bef");
    ASSERT_EQ(s.size(), 3);

    s.erase(1, 10); // Erase more than available
    ASSERT_STREQ(s.cStr(), "b");
    ASSERT_EQ(s.size(), 1);

    s.erase(0, 1); // Erase last char
    ASSERT_STREQ(s.cStr(), "");
    ASSERT_EQ(s.size(), 0);

    String s2("abc");
    s2.erase(5); // Erase out of bounds
    ASSERT_STREQ(s2.cStr(), "abc");
    ASSERT_EQ(s2.size(), 3);
}

TEST_F(StringTest, Find)
{
    String s("hello world");
    ASSERT_EQ(s.find(String("world")), 6);
    ASSERT_EQ(s.find(String("hello")), 0);
    ASSERT_EQ(s.find(String("lo")), 3);
    ASSERT_EQ(s.find(String("foo")), String::npos);
    ASSERT_EQ(s.find(String("o"), 5), 7); // Find 'o' starting from index 5
    ASSERT_EQ(s.find(String("world"), 7),
              String::npos); // 'world' not found from index 7
    ASSERT_EQ(s.find(String("")), String::npos); // Empty string search
    ASSERT_EQ(s.find(String("hello"), s.size()), String::npos);
}

TEST_F(StringTest, Reserve)
{
    String s("short");
    ASSERT_GE(s.capacity(), 5);

    sizeT initial_capacity = s.capacity();
    s.reserve(100);
    ASSERT_GE(s.capacity(), 100);
    ASSERT_STREQ(s.cStr(), "short");
    ASSERT_EQ(s.size(), 5);

    s.reserve(initial_capacity); // Should not shrink
    ASSERT_GE(s.capacity(), 100);
}

TEST_F(StringTest, Append)
{
    String s("beginning");
    char content[] = " middle end";
    s.append(content, std::strlen(content));
    ASSERT_STREQ(s.cStr(), "beginning middle end");
    ASSERT_EQ(s.size(), 20);

    String s2;
    char single_char[] = "a";
    s2.append(single_char, 1);
    ASSERT_STREQ(s2.cStr(), "a");
    ASSERT_EQ(s2.size(), 1);
}

TEST_F(StringTest, OstreamOperator)
{
    String s("test output");
    std::ostringstream oss;
    oss << s;
    ASSERT_EQ(oss.str(), "test output");
}

TEST_F(StringTest, HashSpecialization)
{
    String s1("hashme");
    String s2("hashme");
    String s3("different");
    std::hash<String> hasher;
    ASSERT_EQ(hasher(s1), hasher(s2));
    ASSERT_NE(hasher(s1), hasher(s3));
}

TEST_F(StringTest, SelfAssignment)
{
    String s1("self");
    s1 = s1;
    ASSERT_STREQ(s1.cStr(), "self");

    String s2("move self");
    s2 = std::move(s2);
    ASSERT_STREQ(s2.cStr(), "move self");
}

TEST_F(StringTest, EmptyStringOperations)
{
    String s;
    ASSERT_EQ(s.find(String("a")), String::npos);
    s.erase(0, 1); // Should do nothing
    ASSERT_TRUE(s.empty());
    s.insert(0, 'a');
    ASSERT_STREQ(s.cStr(), "a");
}

TEST_F(StringTest, FindNotFound)
{
    String s("hello");
    ASSERT_EQ(s.find(String("x")), String::npos);
    ASSERT_EQ(s.find(String("helloo")), String::npos);
    ASSERT_EQ(s.find(String("h"), 1), String::npos);
}

TEST_F(StringTest, StartsWithEndsWith)
{
    String s("hello world");
    ASSERT_TRUE(s.startsWith(String("hello")));
    ASSERT_FALSE(s.startsWith(String("world")));
    ASSERT_TRUE(s.endsWith(String("world")));
    ASSERT_FALSE(s.endsWith(String("hello")));

    String s2("h");
    ASSERT_TRUE(s2.startsWith(String("h")));
    ASSERT_TRUE(s2.endsWith(String("h")));

    String empty;
    ASSERT_FALSE(empty.startsWith(String("a")));
    ASSERT_FALSE(empty.endsWith(String("a")));
}
