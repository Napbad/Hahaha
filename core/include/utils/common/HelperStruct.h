// Copyright (c) 2025 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Created: 2025-12-26 10:43:22 by Napbad
//

#ifndef HAHAHA_UTILS_COMMON_HELPER_STRUCT_H
#define HAHAHA_UTILS_COMMON_HELPER_STRUCT_H



#include <initializer_list>
#include <type_traits>

#include "common/definitions.h"

namespace hahaha::util
{

using namespace hahaha::common;

template <typename T> struct isInitList : std::false_type
{
};
template <typename T>
struct isInitList<std::initializer_list<T>> : std::true_type
{
};

template <typename T>
struct isNestedInitList
    : std::integral_constant<
          bool,
          isInitList<T>::value && // is an initializer list
              isInitList<typename T::value_type>::value> // use value_type is no
                                                         // problem
{
};

template <typename T> struct isLegalDataType : std::false_type
{
};

template <> struct isLegalDataType<u8> : std::true_type
{
};

template <> struct isLegalDataType<i8> : std::true_type
{
};

template <> struct isLegalDataType<u16> : std::true_type
{
};

template <> struct isLegalDataType<i16> : std::true_type
{
};

template <> struct isLegalDataType<u32> : std::true_type
{
};

template <> struct isLegalDataType<i32> : std::true_type
{
};

template <> struct isLegalDataType<i64> : std::true_type
{
};

template <> struct isLegalDataType<u64> : std::true_type
{
};

template <> struct isLegalDataType<f32> : std::true_type
{
};

template <> struct isLegalDataType<f64> : std::true_type
{
};


} // namespace hahaha::util

#endif // HAHAHA_UTILS_COMMON_HELPER_STRUCT_H