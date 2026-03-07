#include <gtest/gtest.h>
#include "utils/log/LogColor.h"

using namespace h3::utils;

class LogColorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LogColorTest, ToString_AllColors_ReturnsCorrectAnsiCode) {
    EXPECT_EQ(toString(LogColor::BLACK), "\033[30m");
    EXPECT_EQ(toString(LogColor::RED), "\033[31m");
    EXPECT_EQ(toString(LogColor::GREEN), "\033[32m");
    EXPECT_EQ(toString(LogColor::YELLOW), "\033[33m");
    EXPECT_EQ(toString(LogColor::BLUE), "\033[34m");
    EXPECT_EQ(toString(LogColor::MAGENTA), "\033[35m");
    EXPECT_EQ(toString(LogColor::CYAN), "\033[36m");
    EXPECT_EQ(toString(LogColor::WHITE), "\033[37m");
    EXPECT_EQ(toString(LogColor::RESET), "\033[0m");
}

TEST_F(LogColorTest, ToString_InvalidColor_ReturnsResetCode) {
    // Cast an invalid integer to LogColor to test default case
    EXPECT_EQ(toString(static_cast<LogColor>(999)), "\033[0m");
}
