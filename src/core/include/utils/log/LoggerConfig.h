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

#ifndef HAHAHA_UTILS_LOG_LOGGER_CONFIG_H
#define HAHAHA_UTILS_LOG_LOGGER_CONFIG_H

#include <string>
#include <utility>

#include "utils/log/LogColor.h"
#include "utils/log/LogLevel.h"

namespace hahaha::utils {
constexpr LogColor DefaultColor = LogColor::BLACK;
constexpr LogLevel DefaultLevel = LogLevel::INFO;
constexpr std::string_view DefaultFile = "log.txt";

/**
 * @brief Configuration settings for the Logger.
 */
class LoggerConfig {
  public:
    /**
     * @brief Construct a LoggerConfig with basic settings.
     * @param color Default text color.
     * @param level Minimum log level to display.
     * @param file Output log file path.
     */
    explicit LoggerConfig(LogColor color = DefaultColor,
                          LogLevel level = DefaultLevel,
                          std::string_view file = DefaultFile)
        : color_(color), level_(level), file_(file) {
    }

    /**
     * @brief Construct a full LoggerConfig.
     * @param color Default text color.
     * @param level Minimum log level.
     * @param file Log file path.
     * @param writeToFile Enable file logging.
     * @param writeToConsole Enable console logging.
     * @param timeEnabled Enable timestamping.
     */
    LoggerConfig(LogColor color,
                 LogLevel level,
                 std::string file,
                 bool writeToFile,
                 bool writeToConsole,
                 bool timeEnabled)
        : color_(color), level_(level), file_(std::move(file)),
          writeToFile_(writeToFile), writeToConsole_(writeToConsole),
          timeEnabled_(timeEnabled) {
    }

    [[nodiscard]] LoggerConfig(const LoggerConfig&) = default;
    LoggerConfig(LoggerConfig&&) = default;
    LoggerConfig& operator=(const LoggerConfig&) = default;
    LoggerConfig& operator=(LoggerConfig&&) = default;

    ~LoggerConfig() = default;

    /**
     * @brief Get the default text color.
     * @return LogColor The color setting.
     */
    [[nodiscard]] LogColor getColor() const {
        return color_;
    }

    /**
     * @brief Get the minimum log level.
     * @return LogLevel The log level.
     */
    [[nodiscard]] LogLevel getLevel() const {
        return level_;
    }

    /**
     * @brief Get the log file path.
     * @return std::string_view The file path.
     */
    [[nodiscard]] std::string_view getFile() const {
        return file_;
    }

    /**
     * @brief Check if file logging is enabled.
     * @return bool True if file logging is enabled.
     */
    [[nodiscard]] bool isWriteToFile() const {
        return writeToFile_;
    }

    /**
     * @brief Check if console logging is enabled.
     * @return bool True if console logging is enabled.
     */
    [[nodiscard]] bool isWriteToConsole() const {
        return writeToConsole_;
    }

    /**
     * @brief Check if timestamp is enabled.
     * @return bool True if timestamp is enabled.
     */
    [[nodiscard]] bool isEnableTime() const {
        return timeEnabled_;
    }

    /**
     * @brief Set the default text color.
     * @param color The new color.
     */
    void setColor(LogColor color) {
        color_ = color;
    }

    /**
     * @brief Set the minimum log level.
     * @param level The new log level.
     */
    void setLevel(LogLevel level) {
        level_ = level;
    }

    /**
     * @brief Set the log file path.
     * @param file The new file path.
     */
    void setFile(std::string_view file) {
        file_ = file;
    }

  private:
    LogColor color_;
    LogLevel level_;
    std::string file_;

    bool writeToFile_ = true;
    bool writeToConsole_ = true;
    bool timeEnabled_ = false;
};

} // namespace hahaha::utils

#endif // HAHAHA_UTILS_LOG_LOGGER_CONFIG_H
