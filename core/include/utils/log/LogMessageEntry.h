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

#ifndef HAHAHA_UTILS_LOG_LOG_MESSAGE_ENTRY_H_
#define HAHAHA_UTILS_LOG_LOG_MESSAGE_ENTRY_H_

#include <format>
#include <string>
#include <string_view>

#include "utils/log/LogLevel.h"

namespace hahaha::utils {

/**
 * @brief Represents a single log message entry in the queue.
 *
 * This structure holds the message content and its severity level,
 * and provides a method to format it for output. It is cache-aligned
 * for better performance in the asynchronous logging queue.
 */
struct LogMessageEntry {

  public:
    /**
     * @brief Default constructor. Defaults to INFO level.
     */
    LogMessageEntry() : level_(LogLevel::INFO) {
    }

    /**
     * @brief Construct with level and message.
     * @param level Severity of the message.
     * @param message Content of the message.
     */
    LogMessageEntry(LogLevel level, std::string message)
        : message_(std::move(message)), level_(level) {
    }

    /**
     * @brief Format the entry as a colored string for console output.
     * @return std::string formatted and colored message.
     */
    std::string toString() {
        return std::format("[{}] {}", toColoredString(level_), message_);
    }

    /**
     * @brief Get the raw message string.
     * @return std::string& Reference to the message.
     */
    [[nodiscard]] std::string& getMessage() {
        return message_;
    }

    /**
     * @brief Set the message content.
     * @param message New message string.
     */
    void setMessage(std::string message) {
        message_ = std::move(message);
    }

    /**
     * @brief Get the severity level.
     * @return LogLevel The level of this entry.
     */
    [[nodiscard]] LogLevel getLevel() const {
        return level_;
    }

    /**
     * @brief Set the severity level.
     * @param level New log level.
     */
    void setLevel(LogLevel level) {
        level_ = level;
    }

  private:
    std::string message_; /**< Log message content. */
    LogLevel level_;      /**< Log severity level. */
} __attribute__((aligned(64)));

} // namespace hahaha::utils

#endif // HAHAHA_UTILS_LOG_LOG_MESSAGE_ENTRY_H_
