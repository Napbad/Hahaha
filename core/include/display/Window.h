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

#ifndef HAHAHA_DISPLAY_WINDOW_H
#define HAHAHA_DISPLAY_WINDOW_H

#include "imgui.h"
#include "imgui_internal.h"
#include <string>

namespace hahaha::display {

/**
 * @brief Base Window class for managing ImGui contexts and rendering.
 */
class Window {
  public:
    Window() = default;
    virtual ~Window() = default;

    /**
     * @brief Initialize the window.
     * @param title Window title.
     * @param width Window width.
     * @param height Window height.
     */
    virtual void init(const std::string& title, int width, int height) = 0;

    /**
     * @brief Render a frame.
     * @return bool True if the window should remain open.
     */
    virtual bool render() = 0;

    /**
     * @brief Close the window.
     */
    virtual void close() = 0;
};

} // namespace hahaha::display

#endif // HAHAHA_DISPLAY_WINDOW_H
