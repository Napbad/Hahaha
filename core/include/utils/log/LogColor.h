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

#ifndef HAHAHA_UTILS_LOG_LOG_COLOR_H
#define HAHAHA_UTILS_LOG_LOG_COLOR_H

#include <string_view>
namespace hahaha::utils {

    /**
     * @brief ANSI color codes for console output.
     */
    enum class LogColor {
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        RESET
    };


    /**
     * @brief Convert LogColor to its ANSI escape code.
     * @param color The color.
     * @return std::string_view The ANSI code.
     */
    inline std::string_view toString(LogColor color) { 
        switch (color) {
            case LogColor::BLACK:   return "\033[30m";
            case LogColor::RED:     return "\033[31m";
            case LogColor::GREEN:   return "\033[32m";
            case LogColor::YELLOW:  return "\033[33m";
            case LogColor::BLUE:    return "\033[34m";
            case LogColor::MAGENTA: return "\033[35m";
            case LogColor::CYAN:    return "\033[36m";
            case LogColor::WHITE:   return "\033[37m";
            case LogColor::RESET:   return "\033[0m";
            default:                   return "\033[0m"; // Default to reset if unknown color
        }
    }

}

#endif //HAHAHA_UTILS_LOG_LOG_COLOR_H