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
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
    }

    void TearDown() override {
        if (std::filesystem::exists("test_log.txt")) {
            std::filesystem::remove("test_log.txt");
        }
    }
};

TEST_F(LoggerTest, LoggerConfigAllOptions) {
    LoggerConfig config(
        LogColor::GREEN, LogLevel::DEBUG, "test_log.txt", true, true, true);
    EXPECT_EQ(config.getFile(), "test_log.txt");
    EXPECT_EQ(config.getLevel(), LogLevel::DEBUG);
    EXPECT_TRUE(config.isWriteToFile());
    EXPECT_TRUE(config.isWriteToConsole());
    EXPECT_TRUE(config.isEnableTime());
}

TEST_F(LoggerTest, CustomLoggerFlow) {
    LoggerConfig config(
        LogColor::GREEN, LogLevel::DEBUG, "test_log.txt", true, true, true);
    {
        Logger customLogger(config);
        // Direct call to process is not possible as it's private and running in
        // a thread. We can wait for it to initialize and then it will be
        // destroyed.
    }
    EXPECT_TRUE(std::filesystem::exists("test_log.txt"));
}

TEST_F(LoggerTest, ConvenienceFunctions) {
    // Test all convenience methods in Logger and global scope
    Logger::trace("trace std::string");
    Logger::trace("trace const char*");
    Logger::debug("debug std::string");
    Logger::debug("debug const char*");
    Logger::info("info std::string");
    Logger::info("info const char*");
    Logger::warn("warn std::string");
    Logger::warn("warn const char*");
    Logger::error("error std::string");
    Logger::error("error const char*");
    Logger::fatal("fatal std::string");
    Logger::fatal("fatal const char*");

    // Global convenience functions
    trace("global trace");
    debug("global debug");
    info("global info");
    warn("global warn");
    error("global error");
    fatal("global fatal");
    log("global log", LogLevel::INFO);
    log(std::string("global log string"), LogLevel::INFO);

    // Logger::logWithStacktrace("test stacktrace if available",
    //                                          LogLevel::DEBUG);

    // Give some time for background thread
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(LoggerTest, AllLogLevels) {
    Logger::trace("trace message");
    Logger::debug("debug message");
    Logger::info("info message");
    Logger::warn("warn message");
    Logger::error("error message");
    Logger::fatal("fatal message");

    // Test stacktrace branch
    // Logger::logWithStacktrace("message with stacktrace", LogLevel::ERROR);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_F(LoggerTest, LoggerConfigVariants) {
    // Note: Static methods (Logger::info, Logger::debug, etc.) use the
    // singleton Logger, while custom Logger instances are independent. The
    // singleton Logger's worker thread may still be running after the test, so
    // we need to ensure messages are processed.

    // 1. Both file and console enabled with time
    {
        LoggerConfig config(
            LogColor::RED, LogLevel::INFO, "test_both.txt", true, true, true);
        Logger logger(config);
        Logger::info("Both enabled");
        // Wait for messages to be processed before logger destruction
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // 2. Only console, no time
    {
        LoggerConfig config(
            LogColor::BLUE, LogLevel::DEBUG, "", false, true, false);
        Logger logger(config);
        Logger::debug("Console only, no time");
        // Wait for messages to be processed before logger destruction
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // 3. Only file, with time
    {
        LoggerConfig config(LogColor::GREEN,
                            LogLevel::WARN,
                            "test_file_only.txt",
                            true,
                            false,
                            true);
        Logger logger(config);
        Logger::warn("File only with time");
        // Wait for messages to be processed before logger destruction
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Ensure singleton logger processes any remaining messages
    // The singleton Logger will continue running after the test, which is
    // expected behavior
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (std::filesystem::exists("test_both.txt"))
        std::filesystem::remove("test_both.txt");
    if (std::filesystem::exists("test_file_only.txt"))
        std::filesystem::remove("test_file_only.txt");
}

TEST_F(LoggerTest, ShutdownIdempotency) {
    Logger::shutdown();
    Logger::shutdown(); // Should handle multiple calls
}

TEST_F(LoggerTest, LoggerShutdown) {
    // Test shutdown
    Logger::info("Message before shutdown");
    Logger::shutdown();
    // After shutdown, we shouldn't really call it again, but let's see if it's
    // idempotent or handled The singleton is still there, but worker thread
    // joined.
}

TEST_F(LoggerTest, LogMessageEntrySetters) {
    LogMessageEntry entry;
    entry.setLevel(LogLevel::WARN);
    entry.setMessage("New Message");
    EXPECT_EQ(entry.getLevel(), LogLevel::WARN);
    EXPECT_EQ(entry.getMessage(), "New Message");

    // Test toString() branch
    EXPECT_FALSE(entry.toString().empty());
}

TEST_F(LoggerTest, LoggerWithTimeEnabled) {
    LoggerConfig config(LogColor::CYAN,
                        LogLevel::TRACE,
                        "test_log_time.txt",
                        true,
                        false,
                        true);
    {
        Logger timeLogger(config);
        // Wait briefly for worker to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (std::filesystem::exists("test_log_time.txt")) {
        std::filesystem::remove("test_log_time.txt");
    }
}

TEST_F(LoggerTest, LoggerOutputDisabled) {
    if (std::filesystem::exists("test_log_none.txt")) {
        std::filesystem::remove("test_log_none.txt");
    }
    // Both console and file output disabled
    LoggerConfig config(LogColor::CYAN,
                        LogLevel::TRACE,
                        "test_log_none.txt",
                        false,
                        false,
                        false);
    {
        Logger noneLogger(config);
        // We can't easily check if nothing was written to console, but we check
        // the file.
    }
    EXPECT_FALSE(std::filesystem::exists("test_log_none.txt"));
}
