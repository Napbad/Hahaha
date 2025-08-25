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

#include <iostream>
#include <memory>
#include "include/common/Error.h"
#include "include/common/Res.h"
#include "include/common/util/commonUtil.h"
#include "include/common/ds/str.h"

using namespace hiahiahia;

// Simple function returning Res with automatic type deduction
Res<int, std::string> divide(int a, int b) {
    if (b == 0) {
        return err<std::string>("Division by zero");
    }
    return Ok(a / b);
}

// Function using the ConvertErr error type
Res<double, util::ConvertErr> parseAndMultiply(const ds::Str& input, double factor) {
    // Use the strTo utility
    auto result = util::strTo<double>(input);
    
    // Check if conversion was successful
    if (result.isErr()) {
        return err(result.unwrapErr());
    }
    
    // If successful, multiply and return
    return Ok(result.unwrap() * factor);
}

// Chain calls with andThen
template<typename T>
Res<T, std::string> processValue(Res<T, std::string> res) {
    return res.andThen([](T value) -> Res<T, std::string> {
        if (value < 0) {
            return err<std::string>("Negative values not allowed");
        }
        return Ok(value * 2);
    });
}

int main() {
    // Example 1: Basic usage with Ok and Err
    auto res1 = divide(10, 2);
    if (res1.isOk()) {
        std::cout << "Result: " << res1.unwrap() << std::endl;
    } else {
        std::cout << "Error: " << res1.unwrapErr() << std::endl;
    }

    auto res2 = divide(10, 0);
    if (res2.isOk()) {
        std::cout << "Result: " << res2.unwrap() << std::endl;
    } else {
        std::cout << "Error: " << res2.unwrapErr() << std::endl;
    }

    // Example 2: Using utility function for string to number conversion
    ds::Str validNumber = "123.45";
    ds::Str invalidNumber = "abc";
    
    auto conv1 = parseAndMultiply(validNumber, 2.0);
    if (conv1.isOk()) {
        std::cout << "Conversion result: " << conv1.unwrap() << std::endl;
    } else {
        std::cout << "Conversion error: " << conv1.unwrapErr().toString() << std::endl;
    }
    
    auto conv2 = parseAndMultiply(invalidNumber, 2.0);
    if (conv2.isOk()) {
        std::cout << "Conversion result: " << conv2.unwrap() << std::endl;
    } else {
        std::cout << "Conversion error: " << conv2.unwrapErr().toString() << std::endl;
    }

    // Example 3: Using chain calls
    auto proc1 = processValue(Ok<int, std::string>(10));
    auto proc2 = processValue(Ok<int, std::string>(-5));
    auto proc3 = processValue(err<std::string, int>("Initial error"));

    std::cout << "Process 1: " << (proc1.isOk() ? std::to_string(proc1.unwrap()) : proc1.unwrapErr()) << std::endl;
    std::cout << "Process 2: " << (proc2.isOk() ? std::to_string(proc2.unwrap()) : proc2.unwrapErr()) << std::endl;
    std::cout << "Process 3: " << (proc3.isOk() ? std::to_string(proc3.unwrap()) : proc3.unwrapErr()) << std::endl;

    return 0;
} 