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
// Created: 2025-12-14 22:59:42 by Napbad
// Updated: 2025-12-14 23:02:36
//

#ifndef HAHAHA_COMMON_DEFINITIONS_H
#define HAHAHA_COMMON_DEFINITIONS_H

#include <cstdint>

namespace hahaha::common
{

using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;
using bool8 = std::uint8_t;

static constexpr bool8 True = 1;
static constexpr bool8 False = 0;

} // namespace hahaha::common

#endif // HAHAHA_COMMON_DEFINITIONS_H