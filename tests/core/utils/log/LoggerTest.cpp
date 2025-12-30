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

#include "utils/log/Logger.h"

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>

using hahaha::utils::LogColor;
using hahaha::utils::Logger;
using hahaha::utils::LoggerConfig;
using hahaha::utils::LogLevel;
using hahaha::utils::LogMessageEntry;

class LoggerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Clean up log files before each test
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
        if (std::filesystem::exists("log.txt")) {
            std::filesystem::remove("log.txt");
        }
    }

    void TearDown() override {
        // Clean up log files after each test
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
        if (std::filesystem::exists("log.txt")) {
            std::filesystem::remove("log.txt");
        }
    }
};

TEST_F(LoggerTest, LogLevelToString) {
    EXPECT_EQ(toString(LogLevel::TRACE), "TRACE");
    EXPECT_EQ(toString(LogLevel::DEBUG), "DEBUG");
    EXPECT_EQ(toString(LogLevel::INFO), "INFO ");
    EXPECT_EQ(toString(LogLevel::WARN), "WARN ");
    EXPECT_EQ(toString(LogLevel::ERROR), "ERROR");
    EXPECT_EQ(toString(LogLevel::FATAL), "FATAL");
}

TEST_F(LoggerTest, LoggerColorToString) {
    EXPECT_EQ(toString(LogColor::RED), "\033[31m");
    EXPECT_EQ(toString(LogColor::RESET), "\033[0m");
}

TEST_F(LoggerTest, LoggerConfigDefault) {
    LoggerConfig config;
    EXPECT_EQ(config.getFile(), "log.txt");
    EXPECT_EQ(config.getLevel(), LogLevel::INFO);
    EXPECT_TRUE(config.isWriteToFile());
    EXPECT_TRUE(config.isWriteToConsole());
    EXPECT_FALSE(config.isEnableTime());
}

TEST_F(LoggerTest, CustomLoggerInstance) {
    LoggerConfig config(
        LogColor::GREEN, LogLevel::DEBUG, "test_log.txt", true, false, false);
    {
        Logger customLogger(config);
        // We can't use the static methods as they use the singleton
        // But we can check if the constructor and destructor work
    }
    EXPECT_TRUE(std::filesystem::exists("test_log.txt"));
}

TEST_F(LoggerTest, SingletonLoggerUsage) {
    // This will use the singleton instance, writing to log.txt by default
    hahaha::utils::Logger::info("Test info message");
    hahaha::utils::Logger::error("Test error message");

    // Give some time for the background thread to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check if default log file exists (it might have been created by previous
    // runs or this run) We don't assert its content strictly because it's a
    // singleton and might have other logs
    EXPECT_TRUE(std::filesystem::exists("log.txt"));
}

TEST_F(LoggerTest, LogMessageEntryTest) {
    LogMessageEntry entry(LogLevel::INFO, "Test message");
    EXPECT_EQ(entry.getLevel(), LogLevel::INFO);
    EXPECT_EQ(entry.getMessage(), "Test message");

    std::string str = entry.toString();
    EXPECT_NE(str.find("INFO"), std::string::npos);
    EXPECT_NE(str.find("Test message"), std::string::npos);
}
