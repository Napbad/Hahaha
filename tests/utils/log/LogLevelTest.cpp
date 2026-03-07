#include <gtest/gtest.h>
#include "utils/log/LogLevel.h"

using namespace h3::utils;

class LogLevelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LogLevelTest, ToString_AllLevels_ReturnsCorrectString) {
    EXPECT_EQ(toString(LogLevel::TRACE), "TRACE");
    EXPECT_EQ(toString(LogLevel::DEBUG), "DEBUG");
    EXPECT_EQ(toString(LogLevel::INFO), "INFO ");
    EXPECT_EQ(toString(LogLevel::WARN), "WARN ");
    EXPECT_EQ(toString(LogLevel::ERROR), "ERROR");
    EXPECT_EQ(toString(LogLevel::FATAL), "FATAL");
}

TEST_F(LogLevelTest, ToString_InvalidLevel_ReturnsUnknown) {
    EXPECT_EQ(toString(static_cast<LogLevel>(999)), "UNKNOWN");
}

TEST_F(LogLevelTest, ToColoredString_AllLevels_ReturnsCorrectAnsiString) {
    EXPECT_EQ(toColoredString(LogLevel::TRACE), DefaultColoredTrace);
    EXPECT_EQ(toColoredString(LogLevel::DEBUG), DefaultColoredDebug);
    EXPECT_EQ(toColoredString(LogLevel::INFO), DefaultColoredInfo);
    EXPECT_EQ(toColoredString(LogLevel::WARN), DefaultColoredWarn);
    EXPECT_EQ(toColoredString(LogLevel::ERROR), DefaultColoredError);
    EXPECT_EQ(toColoredString(LogLevel::FATAL), DefaultColoredFatal);
}

TEST_F(LogLevelTest, ToColoredString_InvalidLevel_ReturnsUnknown) {
    EXPECT_EQ(toColoredString(static_cast<LogLevel>(999)), "UNKNOWN");
}
