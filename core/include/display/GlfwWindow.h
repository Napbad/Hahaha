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

#ifndef HAHAHA_DISPLAY_GLFW_WINDOW_H
#define HAHAHA_DISPLAY_GLFW_WINDOW_H

#include <GLFW/glfw3.h>

#include "display/Window.h"

namespace hahaha::display {

/**
 * @brief Concrete Window implementation using GLFW and OpenGL.
 */
class GlfwWindow : public Window {
  public:
    GlfwWindow() = default;
    ~GlfwWindow() override;

    void init(const std::string& title, int width, int height) override;
    bool render() override;
    void close() override;

    [[nodiscard]] GLFWwindow* getHandle() const { return window_; }

  private:
    GLFWwindow* window_ = nullptr;
};

} // namespace hahaha::display

#endif // HAHAHA_DISPLAY_GLFW_WINDOW_H

