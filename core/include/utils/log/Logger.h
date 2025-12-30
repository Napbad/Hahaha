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

#ifndef HAHAHA_UTILS_LOG_LOGGER_H
#define HAHAHA_UTILS_LOG_LOGGER_H

#include <atomic>
#include <condition_variable>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "utils/log/LogLevel.h"
#include "utils/log/LogMessageEntry.h"
#include "utils/log/LoggerConfig.h"

namespace hahaha::utils {
/**
 * @brief Thread-safe asynchronous logging system.
 *
 * Logger uses a background worker thread and a message queue to ensure that
 * logging calls do not block the main execution thread. It supports both
 * console and file output with configurable levels and formatting.
 *
 * Use the static methods (info, warn, error, etc.) for convenient logging.
 */
class Logger {
  public:
    Logger(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&) = delete;

    /**
     * @brief Construct a Logger with specific configuration.
     * @param config The configuration to use.
     */
    explicit Logger(LoggerConfig config)
        : config_(std::move(config)), running_(true) {
        stream_.open(config_.getFile().data());
        workerThread_ = std::thread(&Logger::process, this);
    }

    /**
     * @brief Destructor: flushes the queue and joins the worker thread.
     */
    ~Logger() {
        if (running_) {
            running_ = false;
            condition_.notify_all();
            if (workerThread_.joinable()) {
                workerThread_.join();
            }
        }
        if (stream_.is_open()) {
            stream_.close();
        }
    }

    /**
     * @brief Get the singleton Logger instance.
     * @return Logger& reference to the singleton.
     */
    static Logger& instance() {
        static Logger logger(LoggerConfig{});
        return logger;
    }

    /**
     * @brief Log a message with a specific level.
     * @param msg The message string.
     * @param level Severity level.
     */
    static void log(const std::string& msg, LogLevel level);

    /** @overload log(const char* msg, LogLevel level) */
    static void log(const char* msg, LogLevel level);

    /** @brief Log a FATAL level message. */
    static void fatal(const std::string& msg);
    /** @overload fatal(const char* msg) */
    static void fatal(const char* msg);

    /** @brief Log an ERROR level message. */
    static void error(const std::string& msg);
    /** @overload error(const char* msg) */
    static void error(const char* msg);

    /** @brief Log a WARN level message. */
    static void warn(const std::string& msg);
    /** @overload warn(const char* msg) */
    static void warn(const char* msg);

    /** @brief Log an INFO level message. */
    static void info(const std::string& msg);
    /** @overload info(const char* msg) */
    static void info(const char* msg);

    /** @brief Log a DEBUG level message. */
    static void debug(const std::string& msg);
    /** @overload debug(const char* msg) */
    static void debug(const char* msg);

    /** @brief Log a TRACE level message. */
    static void trace(const std::string& msg);
    /** @overload trace(const char* msg) */
    static void trace(const char* msg);

  private:
    /**
     * @brief Background process that drains the log queue.
     */
    void process() {
        while (running_ || !queue_.empty()) {
            // get log entry
            LogMessageEntry entry;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(
                    lock, [this] { return !queue_.empty() || !running_; });

                if (!running_ && queue_.empty()) {
                    break;
                }

                if (!queue_.empty()) {
                    entry = std::move(queue_.front());
                    queue_.pop();
                } else {
                    continue;
                }
            }

            std::string timestamp;
            getTimeIfEnabled(timestamp);

            if (config_.isWriteToFile()) {
                stream_ << timestamp << "[" << toString(entry.getLevel()) << "]"
                        << entry.getMessage() << '\n';
                stream_.flush(); // Ensure data is written
            }
            if (config_.isWriteToConsole()) {
                std::cout << timestamp << "[" << toString(entry.getLevel())
                          << "]" << entry.getMessage() << '\n';
                std::cout.flush(); // Ensure data is written
            }
        }
    }

    /**
     * @brief Generate timestamp string if enabled in config.
     * @param timestamp Output string for the timestamp.
     */
    void getTimeIfEnabled(std::string& timestamp) {
        if (config_.isEnableTime()) {
            std::time_t now = std::time(nullptr);
            std::tm localTime{};

#ifdef _WIN32
            localtime_s(&localTime, &now); // Windows
#else
            localtime_r(&now, &localTime); // POSIX
#endif

            std::string time =
                std::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}",
                            localTime.tm_year + 1900,
                            localTime.tm_mon + 1,
                            localTime.tm_mday,
                            localTime.tm_hour,
                            localTime.tm_min,
                            localTime.tm_sec);
            timestamp = std::format("[{}]", time);
        }
    }

    LoggerConfig config_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<LogMessageEntry> queue_;
    std::ofstream stream_;
    std::thread workerThread_;
    std::atomic<bool> running_;
};

inline void Logger::log(const std::string& msg, LogLevel level) {
    Logger& logger = instance();
    {
        std::lock_guard<std::mutex> lock(logger.mutex_);
        logger.queue_.emplace(level, msg);
    }
    logger.condition_.notify_one();
}

inline void Logger::log(const char* msg, LogLevel level) {
    log(std::string(msg), level);
}

inline void Logger::fatal(const std::string& msg) {
    log(msg, LogLevel::FATAL);
}

inline void Logger::fatal(const char* msg) {
    log(std::string(msg), LogLevel::FATAL);
}

inline void Logger::error(const std::string& msg) {
    log(msg, LogLevel::ERROR);
}

inline void Logger::error(const char* msg) {
    log(std::string(msg), LogLevel::ERROR);
}

inline void Logger::warn(const std::string& msg) {
    log(msg, LogLevel::WARN);
}

inline void Logger::warn(const char* msg) {
    log(std::string(msg), LogLevel::WARN);
}

inline void Logger::info(const std::string& msg) {
    log(msg, LogLevel::INFO);
}

inline void Logger::info(const char* msg) {
    log(std::string(msg), LogLevel::INFO);
}

inline void Logger::debug(const std::string& msg) {
    log(msg, LogLevel::DEBUG);
}

inline void Logger::debug(const char* msg) {
    log(std::string(msg), LogLevel::DEBUG);
}

inline void Logger::trace(const std::string& msg) {
    log(msg, LogLevel::TRACE);
}

inline void Logger::trace(const char* msg) {
    log(std::string(msg), LogLevel::TRACE);
}

} // namespace hahaha::utils

inline void info(const std::string& msg) {
    hahaha::utils::Logger::info(msg);
}
inline void info(const char* msg) {
    hahaha::utils::Logger::info(msg);
}
inline void debug(const std::string& msg) {
    hahaha::utils::Logger::debug(msg);
}
inline void debug(const char* msg) {
    hahaha::utils::Logger::debug(msg);
}
inline void warn(const std::string& msg) {
    hahaha::utils::Logger::warn(msg);
}
inline void warn(const char* msg) {
    hahaha::utils::Logger::warn(msg);
}
inline void error(const std::string& msg) {
    hahaha::utils::Logger::error(msg);
}
inline void error(const char* msg) {
    hahaha::utils::Logger::error(msg);
}
inline void fatal(const std::string& msg) {
    hahaha::utils::Logger::fatal(msg);
}
inline void fatal(const char* msg) {
    hahaha::utils::Logger::fatal(msg);
}
inline void trace(const std::string& msg) {
    hahaha::utils::Logger::trace(msg);
}
inline void trace(const char* msg) {
    hahaha::utils::Logger::trace(msg);
}
inline void log(const std::string& msg, hahaha::utils::LogLevel level) {
    hahaha::utils::Logger::log(msg, level);
}
inline void log(const char* msg, hahaha::utils::LogLevel level) {
    hahaha::utils::Logger::log(msg, level);
}

#endif // HAHAHA_UTILS_LOG_LOGGER_H