// Copyright (c) 2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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

#include "utils/log/LogColor.h"

#include <gtest/gtest.h>

using hahaha::utils::LogColor;

TEST(LogColorTest, ToString_CoversAllColorsAndDefault) {
    EXPECT_EQ(toString(LogColor::BLACK), "\033[30m");
    EXPECT_EQ(toString(LogColor::RED), "\033[31m");
    EXPECT_EQ(toString(LogColor::GREEN), "\033[32m");
    EXPECT_EQ(toString(LogColor::YELLOW), "\033[33m");
    EXPECT_EQ(toString(LogColor::BLUE), "\033[34m");
    EXPECT_EQ(toString(LogColor::MAGENTA), "\033[35m");
    EXPECT_EQ(toString(LogColor::CYAN), "\033[36m");
    EXPECT_EQ(toString(LogColor::WHITE), "\033[37m");
    EXPECT_EQ(toString(LogColor::RESET), "\033[0m");

    // Force default branch.
    auto invalid = static_cast<LogColor>(999);
    EXPECT_EQ(toString(invalid), "\033[0m");
}
