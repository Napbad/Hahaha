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
// 
// Contributors:
// Napbad (napbad.sen@gmail.com ) (https://github.com/Napbad )
//

#ifndef HAHAHA_UTILS_COMMON_HELPER_STRUCT_H
#define HAHAHA_UTILS_COMMON_HELPER_STRUCT_H



#include <initializer_list>
#include <type_traits>

#include "common/definitions.h"

namespace hahaha::utils
{

using hahaha::common::u32;
using hahaha::common::i32;
using hahaha::common::f32;
using hahaha::common::u64;
using hahaha::common::i64;
using hahaha::common::f64;
using hahaha::common::u8;
using hahaha::common::i8;
using hahaha::common::u16;
using hahaha::common::i16;

/**
 * @brief Type trait to check if a type is std::initializer_list.
 * @tparam T The type to check.
 */
template <typename T> struct isInitList : std::false_type
{
};

/** @brief Specialization for std::initializer_list. */
template <typename T>
struct isInitList<std::initializer_list<T>> : std::true_type
{
};

/**
 * @brief Type trait to check if a type is a nested std::initializer_list (e.g., list of lists).
 * @tparam T The type to check.
 */
template <typename T>
struct isNestedInitList
    : std::integral_constant<
          bool,
          isInitList<T>::value && // is an initializer list
              isInitList<typename T::value_type>::value> // check if element is also a list
{
};

/**
 * @brief Type trait to check if a type is a legally supported data type for Tensors.
 * @tparam T The type to check.
 */
template <typename T> struct isLegalDataType : std::false_type
{
};

/** @brief Specialization for uint8_t. */
template <> struct isLegalDataType<u8> : std::true_type
{
};

/** @brief Specialization for int8_t. */
template <> struct isLegalDataType<i8> : std::true_type
{
};

/** @brief Specialization for uint16_t. */
template <> struct isLegalDataType<u16> : std::true_type
{
};

/** @brief Specialization for int16_t. */
template <> struct isLegalDataType<i16> : std::true_type
{
};

/** @brief Specialization for uint32_t. */
template <> struct isLegalDataType<u32> : std::true_type
{
};

/** @brief Specialization for int32_t. */
template <> struct isLegalDataType<i32> : std::true_type
{
};

/** @brief Specialization for int64_t. */
template <> struct isLegalDataType<i64> : std::true_type
{
};

/** @brief Specialization for uint64_t. */
template <> struct isLegalDataType<u64> : std::true_type
{
};

/** @brief Specialization for float. */
template <> struct isLegalDataType<f32> : std::true_type
{
};

/** @brief Specialization for double. */
template <> struct isLegalDataType<f64> : std::true_type
{
};


} // namespace hahaha::utils

#endif // HAHAHA_UTILS_COMMON_HELPER_STRUCT_H