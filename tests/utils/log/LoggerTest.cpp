#include <gtest/gtest.h>
#include "utils/log/Logger.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace h3::utils;

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state
        if (std::filesystem::exists("log.txt")) {
            std::filesystem::remove("log.txt");
        }
        // Restart logger if it was shut down by previous test
        Logger::restart();
    }

    void TearDown() override {
        // Shutdown to flush and close file
        Logger::shutdown();
        // Clean up
        if (std::filesystem::exists("log.txt")) {
            std::filesystem::remove("log.txt");
        }
    }

    bool fileContains(const std::string& filename, const std::string& content) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.find(content) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

TEST_F(LoggerTest, Singleton_Instance_ReturnsSameInstance) {
    Logger& logger1 = Logger::instance();
    Logger& logger2 = Logger::instance();
    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(LoggerTest, Log_Info_WritesToFile) {
    std::string testMsg = "Test Info Message";
    Logger::info(testMsg);
    
    // Give some time for the worker thread to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // We need to flush or wait. Logger::shutdown() in TearDown will flush.
    // But to check inside the test, we rely on the worker thread loop.
    
    // Force a shutdown to ensure flush for verification (or just wait)
    // Since we can't easily force flush without shutdown, and shutdown stops the thread,
    // we might need to rely on sleep or shutdown.
    
    Logger::shutdown(); // This joins the thread and closes the file.
    
    EXPECT_TRUE(std::filesystem::exists("log.txt"));
    EXPECT_TRUE(fileContains("log.txt", "INFO"));
    EXPECT_TRUE(fileContains("log.txt", testMsg));
}

TEST_F(LoggerTest, Log_AllLevels_WritesToFile) {
    Logger::trace("Trace Msg");
    Logger::debug("Debug Msg");
    Logger::info("Info Msg");
    Logger::warn("Warn Msg");
    Logger::error("Error Msg");
    Logger::fatal("Fatal Msg");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    Logger::shutdown();
    
    EXPECT_TRUE(fileContains("log.txt", "TRACE"));
    EXPECT_TRUE(fileContains("log.txt", "Trace Msg"));
    
    EXPECT_TRUE(fileContains("log.txt", "DEBUG"));
    EXPECT_TRUE(fileContains("log.txt", "Debug Msg"));
    
    EXPECT_TRUE(fileContains("log.txt", "INFO"));
    EXPECT_TRUE(fileContains("log.txt", "Info Msg"));
    
    EXPECT_TRUE(fileContains("log.txt", "WARN"));
    EXPECT_TRUE(fileContains("log.txt", "Warn Msg"));
    
    EXPECT_TRUE(fileContains("log.txt", "ERROR"));
    EXPECT_TRUE(fileContains("log.txt", "Error Msg"));
    
    EXPECT_TRUE(fileContains("log.txt", "FATAL"));
    EXPECT_TRUE(fileContains("log.txt", "Fatal Msg"));
}

TEST_F(LoggerTest, GlobalFunctions_ForwardToLogger) {
    ::info("Global Info");
    ::error("Global Error");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Logger::shutdown();
    
    EXPECT_TRUE(fileContains("log.txt", "Global Info"));
    EXPECT_TRUE(fileContains("log.txt", "Global Error"));
}

TEST_F(LoggerTest, GlobalFunctions_CharOverloads) {
    ::info("Global Char Info");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Logger::shutdown();
    
    EXPECT_TRUE(fileContains("log.txt", "Global Char Info"));
}
