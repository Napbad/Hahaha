// Copyright (c) 2025 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include <filesystem>
#include <gtest/gtest.h>

#include "utils/log/Logger.h"

int main() {


    ::testing::InitGoogleTest();
    int result = RUN_ALL_TESTS();

    // Ensure logger is shut down and file is closed before attempting cleanup
    h3::utils::Logger::shutdown();

    // Clean up default log file if it exists after all tests
    if (std::filesystem::exists("log.txt")) {
        std::filesystem::remove("log.txt");
    }

    return result;
}
