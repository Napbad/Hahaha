#include <gtest/gtest.h>
#include "utils/log/LoggerConfig.h"

using namespace h3::utils;

class LoggerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LoggerConfigTest, DefaultConstructor_InitializesWithDefaults) {
    LoggerConfig config;
    EXPECT_EQ(config.getColor(), DefaultColor);
    EXPECT_EQ(config.getLevel(), DefaultLevel);
    EXPECT_EQ(config.getFile(), DefaultFile);
    EXPECT_TRUE(config.isWriteToFile());
    EXPECT_TRUE(config.isWriteToConsole());
    EXPECT_FALSE(config.isEnableTime());
}

TEST_F(LoggerConfigTest, ParameterizedConstructor_InitializesWithValues) {
    LoggerConfig config(LogColor::RED, LogLevel::ERROR, "test.log");
    EXPECT_EQ(config.getColor(), LogColor::RED);
    EXPECT_EQ(config.getLevel(), LogLevel::ERROR);
    EXPECT_EQ(config.getFile(), "test.log");
    EXPECT_TRUE(config.isWriteToFile());
    EXPECT_TRUE(config.isWriteToConsole());
    EXPECT_FALSE(config.isEnableTime());
}

TEST_F(LoggerConfigTest, FullConstructor_InitializesWithAllValues) {
    LoggerConfig config(LogColor::BLUE, LogLevel::WARN, "custom.log", false, false, true);
    EXPECT_EQ(config.getColor(), LogColor::BLUE);
    EXPECT_EQ(config.getLevel(), LogLevel::WARN);
    EXPECT_EQ(config.getFile(), "custom.log");
    EXPECT_FALSE(config.isWriteToFile());
    EXPECT_FALSE(config.isWriteToConsole());
    EXPECT_TRUE(config.isEnableTime());
}

TEST_F(LoggerConfigTest, Setters_UpdateValuesCorrectly) {
    LoggerConfig config;
    
    config.setColor(LogColor::GREEN);
    EXPECT_EQ(config.getColor(), LogColor::GREEN);
    
    config.setLevel(LogLevel::FATAL);
    EXPECT_EQ(config.getLevel(), LogLevel::FATAL);
    
    config.setFile("new.log");
    EXPECT_EQ(config.getFile(), "new.log");
}

TEST_F(LoggerConfigTest, CopyConstructor_CopiesValuesCorrectly) {
    LoggerConfig config1(LogColor::CYAN, LogLevel::DEBUG, "copy.log", true, false, true);
    LoggerConfig config2(config1);
    
    EXPECT_EQ(config2.getColor(), config1.getColor());
    EXPECT_EQ(config2.getLevel(), config1.getLevel());
    EXPECT_EQ(config2.getFile(), config1.getFile());
    EXPECT_EQ(config2.isWriteToFile(), config1.isWriteToFile());
    EXPECT_EQ(config2.isWriteToConsole(), config1.isWriteToConsole());
    EXPECT_EQ(config2.isEnableTime(), config1.isEnableTime());
}

TEST_F(LoggerConfigTest, MoveConstructor_MovesValuesCorrectly) {
    LoggerConfig config1(LogColor::MAGENTA, LogLevel::TRACE, "move.log", false, true, false);
    LoggerConfig config2(std::move(config1));
    
    EXPECT_EQ(config2.getColor(), LogColor::MAGENTA);
    EXPECT_EQ(config2.getLevel(), LogLevel::TRACE);
    EXPECT_EQ(config2.getFile(), "move.log");
    EXPECT_FALSE(config2.isWriteToFile());
    EXPECT_TRUE(config2.isWriteToConsole());
    EXPECT_FALSE(config2.isEnableTime());

    // Verify state of moved-from object
    // Primitives are copied, so they retain values. std::string is moved, so it should be empty.
    EXPECT_EQ(config1.getFile(), ""); // Moved-from string should be empty
}

TEST_F(LoggerConfigTest, AssignmentOperator_AssignsValuesCorrectly) {
    LoggerConfig config1(LogColor::YELLOW, LogLevel::INFO, "assign.log", true, true, true);
    LoggerConfig config2;
    config2 = config1;
    
    EXPECT_EQ(config2.getColor(), config1.getColor());
    EXPECT_EQ(config2.getLevel(), config1.getLevel());
    EXPECT_EQ(config2.getFile(), config1.getFile());
    EXPECT_EQ(config2.isWriteToFile(), config1.isWriteToFile());
    EXPECT_EQ(config2.isWriteToConsole(), config1.isWriteToConsole());
    EXPECT_EQ(config2.isEnableTime(), config1.isEnableTime());
}

TEST_F(LoggerConfigTest, MoveAssignmentOperator_MovesValuesCorrectly) {
    LoggerConfig config1(LogColor::WHITE, LogLevel::ERROR, "move_assign.log", false, false, false);
    LoggerConfig config2;
    config2 = std::move(config1);
    
    EXPECT_EQ(config2.getColor(), LogColor::WHITE);
    EXPECT_EQ(config2.getLevel(), LogLevel::ERROR);
    EXPECT_EQ(config2.getFile(), "move_assign.log");
    EXPECT_FALSE(config2.isWriteToFile());
    EXPECT_FALSE(config2.isWriteToConsole());
    EXPECT_FALSE(config2.isEnableTime());

    // Verify state of moved-from object
    // Primitives are copied, so they retain values. std::string is moved, so it should be empty.
    EXPECT_EQ(config1.getFile(), ""); // Moved-from string should be empty
}
