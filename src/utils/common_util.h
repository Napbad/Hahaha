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
// Created by Napbad on 8/4/25.
//

#ifndef COMMONUTIL_H
#define COMMONUTIL_H
#include <functional>
#include <utility>

#include "Error.h"
#include "Res.h"

namespace hahaha::core::ds
{
class String;
}
namespace hahaha::core::util
{

/**
 * Helper function to trim whitespace from both enf a string
 * @param s Input string to trim
 * @return Trimmed string
 */
inline String trim(const String& s)
{
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start))
    {
        start++;
    }

    auto end = s.end();
    do
    {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return {start, end + 1};
}


template <typename T>
struct ArithmeticTraits
{
    static constexpr bool isArithmetic = std::is_arithmetic_v<T>;
};

template <typename T>
constexpr bool isArithmetic = ArithmeticTraits<T>::isArithmetic;



} // namespace hahaha::core::util

#endif // COMMONUTIL_H
