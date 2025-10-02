#include "gtest/gtest.h"
#include "common/ds/Str.h"
#include "common/error.h"
#include <sstream>

using namespace hahaha::common::ds;
using namespace hahaha;

// Test fixture for Str
class StrTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Common teardown for tests
    }
};

TEST_F(StrTest, DefaultConstructor) {
    Str s;
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);
    ASSERT_STREQ(s.c_str(), "");
}

TEST_F(StrTest, CStringConstructor) {
    Str s("hello");
    ASSERT_FALSE(s.empty());
    ASSERT_EQ(s.size(), 5);
    ASSERT_STREQ(s.c_str(), "hello");
}

TEST_F(StrTest, CopyConstructor) {
    Str original("world");
    Str copied = original;
    ASSERT_EQ(copied.size(), 5);
    ASSERT_STREQ(copied.c_str(), "world");
    // Ensure deep copy
    copied[0] = 'W';
    ASSERT_STREQ(original.c_str(), "world");
    ASSERT_STREQ(copied.c_str(), "World");
}

TEST_F(StrTest, MoveConstructor) {
    Str original("moveme");
    char* original_data = original.begin();
    Str moved = std::move(original);
    ASSERT_EQ(moved.size(), 6);
    ASSERT_STREQ(moved.c_str(), "moveme");
    ASSERT_TRUE(original.empty());
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(moved.begin(), original_data);
}

TEST_F(StrTest, IteratorConstructor) {
    char arr[] = "test";
    Str s(arr, arr + 4);
    ASSERT_EQ(s.size(), 4);
    ASSERT_STREQ(s.c_str(), "test");
}

TEST_F(StrTest, StdStringConstructor) {
    std::string std_s = "standard";
    Str s(std_s);
    ASSERT_EQ(s.size(), 8);
    ASSERT_STREQ(s.c_str(), "standard");
}

TEST_F(StrTest, StringStreamConstructor) {
    std::stringstream ss;
    ss << "stream";
    Str s(ss);
    ASSERT_EQ(s.size(), 6);
    ASSERT_STREQ(s.c_str(), "stream");
}

TEST_F(StrTest, OStringStreamConstructor) {
    std::ostringstream oss;
    oss << "ostringstream";
    Str s(oss);
    ASSERT_EQ(s.size(), 13);
    ASSERT_STREQ(s.c_str(), "ostringstream");
}

TEST_F(StrTest, Destructor) {
    // Implicitly tested by no crashes on scope exit
    Str* s = new Str("heap");
    delete s;
}

TEST_F(StrTest, CopyAssignment) {
    Str original("source");
    Str assigned;
    assigned = original;
    ASSERT_EQ(assigned.size(), 6);
    ASSERT_STREQ(assigned.c_str(), "source");
    assigned[0] = 'S';
    ASSERT_STREQ(original.c_str(), "source"); // Deep copy
}

TEST_F(StrTest, MoveAssignment) {
    Str original("moving");
    char* original_data = original.begin();
    Str assigned;
    assigned = std::move(original);
    ASSERT_EQ(assigned.size(), 6);
    ASSERT_STREQ(assigned.c_str(), "moving");
    ASSERT_TRUE(original.empty());
    ASSERT_EQ(original.size(), 0);
    ASSERT_EQ(original.begin(), nullptr);
    ASSERT_EQ(assigned.begin(), original_data);
}

TEST_F(StrTest, SizeLengthCapacityEmpty) {
    Str s("12345");
    ASSERT_EQ(s.size(), 5);
    ASSERT_EQ(s.length(), 5);
    ASSERT_GE(s.capacity(), 5);
    ASSERT_FALSE(s.empty());

    Str empty_s;
    ASSERT_TRUE(empty_s.empty());
    ASSERT_EQ(empty_s.size(), 0);
}

TEST_F(StrTest, AtOperator) {
    Str s("abc");
    ASSERT_EQ(s.at(0).unwrap(), 'a');
    ASSERT_EQ(s.at(2).unwrap(), 'c');
    ASSERT_TRUE(s.at(3).isErr());
    ASSERT_EQ(s[0], 'a');
    ASSERT_EQ(s[2], 'c');
}

TEST_F(StrTest, BeginEnd) {
    Str s("abc");
    ASSERT_EQ(*s.begin(), 'a');
    ASSERT_EQ(*(s.end() - 1), 'c');
    ASSERT_EQ(s.end() - s.begin(), s.size());
}

TEST_F(StrTest, Clear) {
    Str s("not empty");
    s.clear();
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);
    ASSERT_STREQ(s.c_str(), "");
}

TEST_F(StrTest, Resize) {
    Str s("abc");
    s.resize(5, 'x');
    ASSERT_EQ(s.size(), 5);
    ASSERT_STREQ(s.c_str(), "abcxx");

    s.resize(2);
    ASSERT_EQ(s.size(), 2);
    ASSERT_STREQ(s.c_str(), "ab");

    Str s2;
    s2.resize(3, 'z');
    ASSERT_EQ(s2.size(), 3);
    ASSERT_STREQ(s2.c_str(), "zzz");
}

TEST_F(StrTest, CStrData) {
    Str s("hello");
    ASSERT_STREQ(s.c_str(), "hello");
    ASSERT_STREQ(s.data(), "hello");
}

TEST_F(StrTest, PlusEqualsOperator) {
    Str s("hello");
    s += " world";
    ASSERT_STREQ(s.c_str(), "hello world");
    ASSERT_EQ(s.size(), 11);

    s += '!';
    ASSERT_STREQ(s.c_str(), "hello world!");
    ASSERT_EQ(s.size(), 12);

    Str s2("foo");
    Str s3("bar");
    s2 += s3;
    ASSERT_STREQ(s2.c_str(), "foobar");
    ASSERT_EQ(s2.size(), 6);
}

TEST_F(StrTest, EqualityOperators) {
    Str s1("test");
    Str s2("test");
    Str s3("Test");
    Str s4("testt");

    ASSERT_TRUE(s1 == s2);
    ASSERT_FALSE(s1 == s3);
    ASSERT_FALSE(s1 == s4);
    ASSERT_TRUE(s1 != s3);
}

TEST_F(StrTest, ComparisonOperators) {
    Str s1("apple");
    Str s2("banana");
    Str s3("apple");

    ASSERT_TRUE(s1 < s2);
    ASSERT_FALSE(s2 < s1);
    ASSERT_TRUE(s1 <= s3);
    ASSERT_TRUE(s2 > s1);
    ASSERT_FALSE(s1 > s2);
    ASSERT_TRUE(s2 >= s1);
}

TEST_F(StrTest, PushBack) {
    Str s("a");
    s.push_back('b');
    ASSERT_STREQ(s.c_str(), "ab");
    ASSERT_EQ(s.size(), 2);
}

TEST_F(StrTest, InsertChar) {
    Str s("ace");
    s.insert(1, 'b');
    ASSERT_STREQ(s.c_str(), "abce");
    ASSERT_EQ(s.size(), 4);

    s.insert(0, 'z');
    ASSERT_STREQ(s.c_str(), "zabce");

    s.insert(s.size(), 'w');
    ASSERT_STREQ(s.c_str(), "zabcew");

    Str s2("hello");
    s2.insert(10, 'x'); // Insert at end if pos > size
    ASSERT_STREQ(s2.c_str(), "hellox");
}

TEST_F(StrTest, InsertStr) {
    Str s("ac");
    Str to_insert("bb");
    s.insert(1, to_insert);
    ASSERT_STREQ(s.c_str(), "abbc");
    ASSERT_EQ(s.size(), 4);

    s.insert(0, Str("xx"));
    ASSERT_STREQ(s.c_str(), "xxabbc");

    s.insert(s.size(), Str("yy"));
    ASSERT_STREQ(s.c_str(), "xxabbcyy");

    Str s2("test");
    s2.insert(10, Str("end")); // Insert at end if pos > size
    ASSERT_STREQ(s2.c_str(), "testend");

    Str s3("initial");
    s3.insert(3, Str("")); // Inserting empty string should not change
    ASSERT_STREQ(s3.c_str(), "initial");
}

TEST_F(StrTest, Erase) {
    Str s("abcdefg");
    s.erase(2, 2); // Remove cd
    ASSERT_STREQ(s.c_str(), "abefg");
    ASSERT_EQ(s.size(), 5);

    s.erase(0, 1); // Remove a
    ASSERT_STREQ(s.c_str(), "befg");
    ASSERT_EQ(s.size(), 4);

    s.erase(3); // Remove g (default count 1)
    ASSERT_STREQ(s.c_str(), "bef");
    ASSERT_EQ(s.size(), 3);

    s.erase(1, 10); // Erase more than available
    ASSERT_STREQ(s.c_str(), "b");
    ASSERT_EQ(s.size(), 1);

    s.erase(0, 1); // Erase last char
    ASSERT_STREQ(s.c_str(), "");
    ASSERT_EQ(s.size(), 0);

    Str s2("abc");
    s2.erase(5); // Erase out of bounds
    ASSERT_STREQ(s2.c_str(), "abc");
    ASSERT_EQ(s2.size(), 3);
}

TEST_F(StrTest, Find) {
    Str s("hello world");
    ASSERT_EQ(s.find(Str("world")), 6);
    ASSERT_EQ(s.find(Str("hello")), 0);
    ASSERT_EQ(s.find(Str("lo")), 3);
    ASSERT_EQ(s.find(Str("foo")), Str::npos);
    ASSERT_EQ(s.find(Str("o"), 5), 7); // Find 'o' starting from index 5
    ASSERT_EQ(s.find(Str("world"), 7), Str::npos); // 'world' not found from index 7
    ASSERT_EQ(s.find(Str("")), Str::npos); // Empty string search
    ASSERT_EQ(s.find(Str("hello"), s.size()), Str::npos);
}

TEST_F(StrTest, Reserve) {
    Str s("short");
    ASSERT_GE(s.capacity(), 5);

    sizeT initial_capacity = s.capacity();
    s.reserve(100);
    ASSERT_GE(s.capacity(), 100);
    ASSERT_STREQ(s.c_str(), "short");
    ASSERT_EQ(s.size(), 5);

    s.reserve(initial_capacity); // Should not shrink
    ASSERT_GE(s.capacity(), 100);
}

TEST_F(StrTest, Append) {
    Str s("beginning");
    char content[] = " middle end";
    s.append(content, std::strlen(content));
    ASSERT_STREQ(s.c_str(), "beginning middle end");
    ASSERT_EQ(s.size(), 20);

    Str s2;
    char single_char[] = "a";
    s2.append(single_char, 1);
    ASSERT_STREQ(s2.c_str(), "a");
    ASSERT_EQ(s2.size(), 1);
}

TEST_F(StrTest, OstreamOperator) {
    Str s("test output");
    std::ostringstream oss;
    oss << s;
    ASSERT_EQ(oss.str(), "test output");
}

TEST_F(StrTest, HashSpecialization) {
    Str s1("hashme");
    Str s2("hashme");
    Str s3("different");
    std::hash<Str> hasher;
    ASSERT_EQ(hasher(s1), hasher(s2));
    ASSERT_NE(hasher(s1), hasher(s3));
} 