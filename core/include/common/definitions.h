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

#ifndef HAHAHA_COMMON_DEFINITIONS_H
#define HAHAHA_COMMON_DEFINITIONS_H

#include <cstdint>

/**
 * @brief Common type aliases and constants for the Hahaha library.
 */
namespace hahaha::common {

using i8 = std::int8_t;     /**< 8-bit signed integer. */
using u8 = std::uint8_t;    /**< 8-bit unsigned integer. */
using i16 = std::int16_t;   /**< 16-bit signed integer. */
using u16 = std::uint16_t;  /**< 16-bit unsigned integer. */
using i32 = std::int32_t;   /**< 32-bit signed integer. */
using u32 = std::uint32_t;  /**< 32-bit unsigned integer. */
using i64 = std::int64_t;   /**< 64-bit signed integer. */
using u64 = std::uint64_t;  /**< 64-bit unsigned integer. */
using f32 = float;          /**< 32-bit floating point. */
using f64 = double;         /**< 64-bit floating point. */
using bool8 = std::uint8_t; /**< 8-bit boolean representation. */

static constexpr bool8 True = 1;  /**< Boolean true. */
static constexpr bool8 False = 0; /**< Boolean false. */

} // namespace hahaha::common

#endif // HAHAHA_COMMON_DEFINITIONS_H