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
// Created by Napbad on 7/12/25.
//

#include <gtest/gtest.h>
#include <common/Res.h>
#include <common/Error.h>
#include <string>
#include <memory>

namespace hahaha {
namespace test {

// Custom error type for testing
class TestError : public Error {
private:
    std::string msg;
    std::string loc;

public:
    explicit TestError(std::string message, std::string location = "test") 
        : msg(std::move(message)), loc(std::move(location)) {}

    [[nodiscard]] ds::Str typeName() const override { return ds::Str("TestError"); }
    [[nodiscard]] ds::Str message() const override { return ds::Str(msg.c_str()); }
    [[nodiscard]] ds::Str location() const override { return ds::Str(loc.c_str()); }
    [[nodiscard]] ds::Str toString() const override { 
        return typeName() + ds::Str(": ") + message() + ds::Str(" at ") + location(); 
    }
    
    bool operator==(const TestError& other) const {
        return msg == other.msg && loc == other.loc;
    }
};

class ResTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test construction of Ok values
TEST_F(ResTest, OkConstruction) {
    // Using value directly
    Res<int, TestError> r1 = Res<int, TestError>::ok(42);
    EXPECT_TRUE(r1.isOk());
    EXPECT_FALSE(r1.isErr());
    EXPECT_EQ(r1.unwrap(), 42);

    // Using lvalue
    int val = 42;
    Res<int, TestError> r2 = Res<int, TestError>::ok(val);
    EXPECT_TRUE(r2.isOk());
    EXPECT_FALSE(r2.isErr());
    EXPECT_EQ(r2.unwrap(), 42);
    
    // Copy construction
    Res<int, TestError> r3 = r1;
    EXPECT_TRUE(r3.isOk());
    EXPECT_FALSE(r3.isErr());
    EXPECT_EQ(r3.unwrap(), 42);
    
    // Move construction
    Res<int, TestError> r4 = std::move(r2);
    EXPECT_TRUE(r4.isOk());
    EXPECT_FALSE(r4.isErr());
    EXPECT_EQ(r4.unwrap(), 42);
}

// Test construction of Err values
TEST_F(ResTest, ErrConstruction) {
    // Using value directly
    Res<int, TestError> r1 = Res<int, TestError>::err(TestError("error1"));
    EXPECT_FALSE(r1.isOk());
    EXPECT_TRUE(r1.isErr());
    EXPECT_EQ(r1.unwrapErr().message().data(), std::string("error1"));

    // Using lvalue
    TestError err("error2");
    Res<int, TestError> r2 = Res<int, TestError>::err(err);
    EXPECT_FALSE(r2.isOk());
    EXPECT_TRUE(r2.isErr());
    EXPECT_EQ(r2.unwrapErr().message().data(), std::string("error2"));
    
    // Using raw pointer
    TestError* errPtr = new TestError("error3");
    Res<int, TestError> r3 = Res<int, TestError>::err(errPtr);
    EXPECT_FALSE(r3.isOk());
    EXPECT_TRUE(r3.isErr());
    EXPECT_EQ(r3.unwrapErr().message().data(), std::string("error3"));
    
    // Using unique_ptr
    auto errUniq = std::make_unique<TestError>("error4");
    Res<int, TestError> r4 = Res<int, TestError>::err(std::move(errUniq));
    EXPECT_FALSE(r4.isOk());
    EXPECT_TRUE(r4.isErr());
    EXPECT_EQ(r4.unwrapErr().message().data(), std::string("error4"));
}

// Test unwrap methods
TEST_F(ResTest, Unwrap) {
    Res<int, TestError> ok = Res<int, TestError>::ok(42);
    Res<int, TestError> err = Res<int, TestError>::err(TestError("error"));
    
    // Test normal unwrap
    EXPECT_EQ(ok.unwrap(), 42);
    EXPECT_THROW(err.unwrap(), std::runtime_error);
    
    // Test const unwrap
    const Res<int, TestError>& constOk = ok;
    EXPECT_EQ(constOk.unwrap(), 42);
    
    // Test move unwrap (should consume the value)
    EXPECT_EQ(std::move(ok).unwrap(), 42);
}

// Test unwrapErr methods
TEST_F(ResTest, UnwrapErr) {
    Res<int, TestError> ok = Res<int, TestError>::ok(42);
    Res<int, TestError> err = Res<int, TestError>::err(TestError("error"));
    
    // Test normal unwrapErr
    EXPECT_THROW(ok.unwrapErr(), std::runtime_error);
    EXPECT_EQ(err.unwrapErr().message().data(), std::string("error"));
    
    // Test const unwrapErr
    const Res<int, TestError>& constErr = err;
    EXPECT_EQ(constErr.unwrapErr().message().data(), std::string("error"));
    
    // Test move unwrapErr (should consume the error)
    TestError movedErr = std::move(err).unwrapErr();
    EXPECT_EQ(movedErr.message().data(), std::string("error"));
}

// Test map method
TEST_F(ResTest, Map) {
    Res<int, TestError> ok = Res<int, TestError>::ok(42);
    Res<int, TestError> err = Res<int, TestError>::err(TestError("error"));
    
    // Map on Ok value
    auto mappedOk = ok.map([](const int& val) { return val * 2; });
    EXPECT_TRUE(mappedOk.isOk());
    EXPECT_EQ(mappedOk.unwrap(), 84);
    
    // Map on Err value
    auto mappedErr = err.map([](const int& val) { return val * 2; });
    EXPECT_TRUE(mappedErr.isErr());
    EXPECT_EQ(mappedErr.unwrapErr().message().data(), std::string("error"));
    
    // Move semantics with map
    auto movedMap = std::move(ok).map([](int&& val) { return val * 3; });
    EXPECT_TRUE(movedMap.isOk());
    EXPECT_EQ(movedMap.unwrap(), 126);
}

// Test mapErr method
TEST_F(ResTest, MapErr) {
    Res<int, TestError> ok = Res<int, TestError>::ok(42);
    Res<int, TestError> err = Res<int, TestError>::err(TestError("error"));
    
    // MapErr on Ok value
    auto mappedOk = ok.mapErr([](const TestError& e) { 
        return TestError(std::string(e.message().data()) + "-mapped"); 
    });
    EXPECT_TRUE(mappedOk.isOk());
    EXPECT_EQ(mappedOk.unwrap(), 42);
    
    // MapErr on Err value
    auto mappedErr = err.mapErr([](const TestError& e) { 
        return TestError(std::string(e.message().data()) + "-mapped"); 
    });
    EXPECT_TRUE(mappedErr.isErr());
    EXPECT_EQ(mappedErr.unwrapErr().message().data(), std::string("error-mapped"));
    
    // Move semantics with mapErr
    auto movedMap = std::move(err).mapErr([](TestError&& e) {
        return TestError(std::string(e.message().data()) + "-moved");
    });
    EXPECT_TRUE(movedMap.isErr());
    EXPECT_EQ(movedMap.unwrapErr().message().data(), std::string("error-moved"));
}

// Test the macros
TEST_F(ResTest, Macros) {
    // Test function using macros
    auto testFunc = [](bool succeed) -> Res<int, TestError> {
        SetRetT(int, TestError);
        if (succeed) {
            Ok(42);
        } else {
            Err(newE(TestError, "macro-error"));
        }
    };
    
    // Test Ok case
    auto okResult = testFunc(true);
    EXPECT_TRUE(okResult.isOk());
    EXPECT_EQ(okResult.unwrap(), 42);
    
    // Test Err case
    auto errResult = testFunc(false);
    EXPECT_TRUE(errResult.isErr());
    EXPECT_EQ(errResult.unwrapErr().message().data(), std::string("macro-error"));
}

// Test with complex types
TEST_F(ResTest, ComplexTypes) {
    // Using string as the value type
    Res<std::string, TestError> strRes = Res<std::string, TestError>::ok(std::string("hello"));
    EXPECT_TRUE(strRes.isOk());
    EXPECT_EQ(strRes.unwrap(), "hello");
    
    // Using unique_ptr as the value type
    auto ptr = std::make_unique<int>(42);
    Res<std::unique_ptr<int>, TestError> ptrRes = Res<std::unique_ptr<int>, TestError>::ok(std::move(ptr));
    EXPECT_TRUE(ptrRes.isOk());
    EXPECT_EQ(*ptrRes.unwrap(), 42);
}

// Test transforming types with map
TEST_F(ResTest, TypeTransformation) {
    Res<int, TestError> intRes = Res<int, TestError>::ok(42);
    
    // Transform int to string
    auto stringRes = intRes.map([](const int& val) { return std::to_string(val); });
    EXPECT_TRUE(stringRes.isOk());
    EXPECT_EQ(stringRes.unwrap(), "42");
    
    // Transform error type
    Res<int, TestError> errRes = Res<int, TestError>::err(TestError("old-error"));
    auto newErrRes = errRes.mapErr([](const TestError& e) {
        return TestError(std::string(e.message().data()) + "-transformed", "new-location");
    });
    EXPECT_TRUE(newErrRes.isErr());
    EXPECT_EQ(newErrRes.unwrapErr().message().data(), std::string("old-error-transformed"));
    EXPECT_EQ(newErrRes.unwrapErr().location().data(), std::string("new-location"));
}

} // namespace test
} // namespace hahaha