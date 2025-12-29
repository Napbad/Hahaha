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

#ifndef HAHAHA_UTILS_LOG_LOGGER_LEVEL_H
#define HAHAHA_UTILS_LOG_LOGGER_LEVEL_H

#include <string_view>

#include "utils/log/LogColor.h"
namespace hahaha::utils
{

/**
 * @brief Severity levels for log messages.
 */
enum class LogLevel
{
    TRACE, /**< Fine-grained informational events. */
    DEBUG, /**< Detailed information useful for debugging. */
    INFO,  /**< General informational messages about progress. */
    WARN,  /**< Potentially harmful situations. */
    ERROR, /**< Error events that might still allow the application to continue. */
    FATAL  /**< Very severe error events that will presumably lead the application to abort. */
};

/**
 * @brief Convert LogLevel to a plain string.
 * @param level The log level.
 * @return std::string_view textual representation (e.g. "INFO ").
 */
inline std::string_view toString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::TRACE:
        return "TRACE";
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO ";
    case LogLevel::WARN:
        return "WARN ";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::FATAL:
        return "FATAL";
    }
}

/** ANSI escape codes for colored log levels in console. */
const std::string_view DefaultColoredDebug = "\033[0;36mDEBUG\033[0m";  // CYAN
const std::string_view DefaultColoredInfo = "\033[0;32mINFO \033[0m";    // GREEN
const std::string_view DefaultColoredWarn = "\033[0;33mWARN \033[0m";    // YELLOW
const std::string_view DefaultColoredError = "\033[0;31mERROR\033[0m";  // RED
const std::string_view DefaultColoredFatal = "\033[0;35mFATAL\033[0m";  // MAGENTA
const std::string_view DefaultColoredTrace = "\033[0;34mTRACE\033[0m";  // BLUE


/**
 * @brief Convert LogLevel to a colored string for console output.
 * @param level The log level.
 * @return std::string_view ANSI colored representation.
 */
inline std::string_view toColoredString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::TRACE:
        return DefaultColoredTrace;
    case LogLevel::DEBUG:
        return DefaultColoredDebug;
    case LogLevel::INFO:
        return DefaultColoredInfo;
    case LogLevel::WARN:
        return DefaultColoredWarn;
    case LogLevel::ERROR:
        return DefaultColoredError;
    case LogLevel::FATAL:
        return DefaultColoredFatal;
    }
}


} // namespace hahaha::utils

#endif // HAHAHA_UTILS_LOG_LOGGER_LEVEL_H