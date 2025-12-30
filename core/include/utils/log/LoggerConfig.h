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

    [[nodiscard]] LogColor getColor() const {
        return color_;
    }

    [[nodiscard]] LogLevel getLevel() const {
        return level_;
    }

    [[nodiscard]] std::string_view getFile() const {
        return file_;
    }

    [[nodiscard]] bool isWriteToFile() const {
        return writeToFile_;
    }

    [[nodiscard]] bool isWriteToConsole() const {
        return writeToConsole_;
    }

    [[nodiscard]] bool isEnableTime() const {
        return timeEnabled_;
    }

    void setColor(LogColor color) {
        color_ = color;
    }

    void setLevel(LogLevel level) {
        level_ = level;
    }

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