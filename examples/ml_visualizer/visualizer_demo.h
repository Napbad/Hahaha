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

#ifndef HAHAHA_EXAMPLE_ML_VISUALIZER_DEMO_H
#define HAHAHA_EXAMPLE_ML_VISUALIZER_DEMO_H

#include "display/Visualizer.h"
#include "display/GlfwWindow.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include <GL/gl.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

namespace hahaha::example {

static int frameCount = 0;

/**
 * @brief Simple demo showing how to use the MLVisualizer with a real GLFW window.
 */
inline void run_visualizer_demo() {
    using namespace hahaha::display;
    
    // 1. Create and initialize the window (handles GLFW/OpenGL/ImGui initialization)
    GlfwWindow window;
    try {
        window.init("Hahaha ML Training Demo", 1280, 720);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize window: " << e.what() << std::endl;
        return;
    }
    
    // 2. Create the visualizer
    auto visualizer = createMLVisualizer();
    
    // 3. Define our "Mock" model structure
    visualizer->addLayer({"Input (Images)", 784, 256});
    visualizer->addLayer({"ReLU", 256, 256});
    visualizer->addLayer({"Hidden 1", 256, 128});
    visualizer->addLayer({"ReLU", 128, 128});
    visualizer->addLayer({"Hidden 2", 128, 64});
    visualizer->addLayer({"Sigmoid", 64, 64});
    visualizer->addLayer({"Output (Classes)", 64, 10});
    
    visualizer->setStatus("Ready to train");
    
    int epoch = 0;
    bool training = true;
    float currentLoss = 2.5f;
    float currentAcc = 0.1f;

    std::cout << "Starting Visualizer loop with a real GLFW window." << std::endl;

    // 4. Main Loop
    while (window.render()) {
        
        // Handle User Interaction
        ControlAction action = visualizer->getControlAction();
        if (action == ControlAction::Start) {
            training = true;
            visualizer->setStatus("Training in progress...");
        } else if (action == ControlAction::Pause) {
            training = false;
            visualizer->setStatus("Training paused");
        } else if (action == ControlAction::Stop) {
            std::cout << "User requested stop via UI." << std::endl;
            break; 
        } else if (action == ControlAction::Reset) {
            epoch = 0;
            currentLoss = 2.5f;
            currentAcc = 0.1f;
            visualizer->setStatus("Reset completed");
        }

        // Simulate Training Step
        if (training) {
            currentLoss *= 0.995f; 
            currentAcc = 1.0f - (currentLoss / 2.5f);
            
            if (epoch < 1000 && (frameCount % 10 == 0)) {
                visualizer->recordMetrics(epoch++, currentLoss, currentAcc);
                if (epoch % 10 == 0) {
                    std::cout << "[Demo] Epoch: " << epoch << ", Loss: " << currentLoss << ", Acc: " << currentAcc << std::endl;
                }
            }
        }
        frameCount++;

        // Render the UI widgets
        visualizer->renderFrame();

        // Finalize rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window.getHandle(), &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window.getHandle());

        // Small sleep to control frame rate if vsync is off or for simulation stability
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace hahaha::example

#endif // HAHAHA_EXAMPLE_ML_VISUALIZER_DEMO_H
