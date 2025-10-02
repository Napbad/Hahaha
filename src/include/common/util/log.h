// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

//
// Created by Napbad on 8/1/25.
//

#ifndef LOG_H
#define LOG_H
#include "common/ds/Str.h"
#include "common/ds/Vec.h"
#include "common/ds/queue.h"
#include <condition_variable>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace hahaha::common::util {
    enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

    static auto _globalLevel = LogLevel::INFO;

    inline ds::Str logLevelToStr(const LogLevel level) {
        switch (level) {
        case LogLevel::DEBUG:
            return ds::Str("DEBUG");
        case LogLevel::INFO:
            return ds::Str("INFO");
        case LogLevel::WARNING:
            return ds::Str("WARNING");
        case LogLevel::ERROR:
            return ds::Str("ERROR");
        case LogLevel::FATAL:
            return ds::Str("FATAL");
        default:
            return ds::Str("UNKNOWN");
        }
    }

    enum class TimeFormat {
        ISO8601, // 2023-12-07T10:30:45Z
        HUMAN_READABLE, // 2023-12-07 10:30:45
        TIME_ONLY, // 10:30:45
        DATE_ONLY // 2023-12-07
    };

    inline ds::Str formatTimePoint(
        const std::chrono::system_clock::time_point& tp, TimeFormat format = TimeFormat::HUMAN_READABLE) {
        const auto time  = std::chrono::system_clock::to_time_t(tp);
        const std::tm tm = *std::localtime(&time);

        std::ostringstream oss;

        switch (format) {
        case TimeFormat::ISO8601:
            oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
            break;
        case TimeFormat::HUMAN_READABLE:
            oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            break;
        case TimeFormat::TIME_ONLY:
            oss << std::put_time(&tm, "%H:%M:%S");
            break;
        case TimeFormat::DATE_ONLY:
            oss << std::put_time(&tm, "%Y-%m-%d");
            break;
        }

        return ds::Str(oss);
    }

    struct LogEntry {
        LogLevel level;
        ds::Str message;
        std::thread::id tid;
        std::chrono::system_clock::time_point timestamp;
        std::string file;
        uint line;
        [[nodiscard]] ds::Str toStr() const {
            return ds::Str("[") + logLevelToStr(level) + "]" + "(" + formatTimePoint(timestamp) + ")" + message + "\n";
        }
    };

    class LogOutput {
    public:
        virtual ~LogOutput()                      = default;
        virtual void write(const LogEntry& entry) = 0;
    };

    class FileOutput final : public LogOutput {
    public:
        explicit FileOutput(const ds::Str& filename) {
            _file.open(filename.data(), std::ios::app);
        }

        ~FileOutput() override {
            if (_file.is_open()) {
                _file.close();
            }
        }

        void write(const LogEntry& entry) override {
            if (_file.is_open()) {
                _file << entry.toStr() << std::endl;
            }
        }

    private:
        std::ofstream _file;
    };

    class ConsoleOutput final : public LogOutput {
    public:
        void write(const LogEntry& entry) override {
            std::cout << entry.toStr() << std::endl;
        }
    };

    class Logger {
    public:
        const ds::Str ConsoleOutputName = ds::Str("console");

        Logger() {
            addOutput(ConsoleOutputName);
            _messageQueue = {};
            startWorkerThread();
        }

        ~Logger() {
            stopWorkerThread();
            for (const auto& output : _outputs) {
                delete output;
            }
        }

        void log(const LogEntry& logMsg) {
            {
                std::lock_guard lk(_queueMutex);
                _messageQueue.push(logMsg);
            }
            _queueCond.notify_one();
        }

        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }

        // Set global log level
        static void setLogLevel(LogLevel level) {
            _globalLevel = level;
        }

        void addOutput(const ds::Str& fileName) {
            if (fileName == ConsoleOutputName) {
                _outputs.emplace_back(new ConsoleOutput());
            } else {
                _outputs.emplace_back(new FileOutput(fileName));
            }
        }

    private:
        ds::Vec<LogOutput*> _outputs;

        ds::queue<LogEntry> _messageQueue;
        mutable std::mutex _queueMutex;
        std::condition_variable _queueCond{};
        std::thread _workerThread;
        std::atomic<bool> running_ = false;

        void startWorkerThread() {
            running_      = true;
            _workerThread = std::thread(&Logger::processQueue, this);
        }

        void stopWorkerThread() {
            if (!running_.load()) {
                return;
            }

            running_ = false;
            _queueCond.notify_all();
            if (_workerThread.joinable()) {
                _workerThread.join();
            }
        }

        void processQueue() {
            while (running_.load()) {
                std::unique_lock lock(_queueMutex);
                _queueCond.wait(lock, [this] { return !_messageQueue.empty() || !running_.load(); });

                if (!running_.load()) {
                    break;
                }

                while (!_messageQueue.empty()) {
                    auto entry = _messageQueue.front();
                    for (auto output : _outputs) {
                        output->write(entry);
                    }
                    _messageQueue.pop();
                }
            }
        }
    };

#define info(msg)                                                                                \
    Logger::getInstance().log(LogEntry{LogLevel::INFO, ds::Str(msg), std::this_thread::get_id(), \
        std::chrono::system_clock::now(), __FILE__, __LINE__})
#define debug(msg)                                                                                \
    Logger::getInstance().log(LogEntry{LogLevel::DEBUG, ds::Str(msg), std::this_thread::get_id(), \
        std::chrono::system_clock::now(), __FILE__, __LINE__})
#define warning(msg)                                                                                \
    Logger::getInstance().log(LogEntry{LogLevel::WARNING, ds::Str(msg), std::this_thread::get_id(), \
        std::chrono::system_clock::now(), __FILE__, __LINE__})
#define error(msg)                                                                                \
    Logger::getInstance().log(LogEntry{LogLevel::ERROR, ds::Str(msg), std::this_thread::get_id(), \
        std::chrono::system_clock::now(), __FILE__, __LINE__})
#define fatal(msg)                                                                                \
    Logger::getInstance().log(LogEntry{LogLevel::FATAL, ds::Str(msg), std::this_thread::get_id(), \
        std::chrono::system_clock::now(), __FILE__, __LINE__})


} // namespace hahaha::common::util

#endif // LOG_H
