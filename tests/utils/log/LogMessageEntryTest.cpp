#include <gtest/gtest.h>
#include "utils/log/LogMessageEntry.h"

using namespace h3::utils;

class LogMessageEntryTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LogMessageEntryTest, DefaultConstructor_InitializesWithDefaults) {
    LogMessageEntry entry;
    EXPECT_EQ(entry.getLevel(), LogLevel::INFO);
    EXPECT_EQ(entry.getMessage(), "");
}

TEST_F(LogMessageEntryTest, ParameterizedConstructor_InitializesWithValues) {
    LogMessageEntry entry(LogLevel::ERROR, "Test Error");
    EXPECT_EQ(entry.getLevel(), LogLevel::ERROR);
    EXPECT_EQ(entry.getMessage(), "Test Error");
}

TEST_F(LogMessageEntryTest, Setters_UpdateValuesCorrectly) {
    LogMessageEntry entry;
    
    entry.setLevel(LogLevel::WARN);
    EXPECT_EQ(entry.getLevel(), LogLevel::WARN);
    
    entry.setMessage("New Message");
    EXPECT_EQ(entry.getMessage(), "New Message");
}

TEST_F(LogMessageEntryTest, ToString_FormatsCorrectly) {
    LogMessageEntry entry(LogLevel::INFO, "Test Info");
    std::string expected = std::format("[{}] {}", toColoredString(LogLevel::INFO), "Test Info");
    EXPECT_EQ(entry.toString(), expected);
}

TEST_F(LogMessageEntryTest, GetMessage_ReturnsReference) {
    LogMessageEntry entry(LogLevel::DEBUG, "Debug Msg");
    std::string& msg = entry.getMessage();
    msg = "Modified Msg";
    EXPECT_EQ(entry.getMessage(), "Modified Msg");
}
