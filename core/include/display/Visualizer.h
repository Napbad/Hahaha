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

#ifndef HAHAHA_DISPLAY_VISUALIZER_H
#define HAHAHA_DISPLAY_VISUALIZER_H

#include <memory>
#include <string>
#include <vector>

#include "math/TensorWrapper.h"

namespace hahaha::display {

/**
 * @brief Represents a layer in the visual graph.
 */
struct LayerInfo {
    std::string name;
    int inputSize;
    int outputSize;
};

/**
 * @brief User actions from the control panel.
 */
enum class ControlAction { None, Start, Pause, Stop, Reset };

/**
 * @brief Top-level interface for visualizing Machine Learning models and
 * training progress.
 */
class IVisualizer {
  public:
    virtual ~IVisualizer() = default;

    /**
     * @brief Initialize the visualization window/context.
     * @param title Title of the visualization window.
     * @param width Width of the window.
     * @param height Height of the window.
     */
    virtual void
    init(const std::string& title, int width = 1280, int height = 720) = 0;

    /**
     * @brief Clear the current model structure visualization.
     */
    virtual void clearModel() = 0;

    /**
     * @brief Add a layer to the model visualization.
     * @param info Information about the layer.
     */
    virtual void addLayer(const LayerInfo& info) = 0;

    /**
     * @brief Record training metrics for a single step/epoch.
     * @param epoch Current epoch index.
     * @param loss Current loss value.
     * @param accuracy Current accuracy value (optional).
     */
    virtual void
    recordMetrics(int epoch, float loss, float accuracy = 0.0f) = 0;

    /**
     * @brief Visualize a tensor as an image/heatmap.
     * @param name Name of the visualization window.
     * @param data Tensor data to visualize.
     */
    virtual void
    visualizeTensor(const std::string& name,
                    const hahaha::math::TensorWrapper<float>& data) = 0;

    /**
     * @brief Render the current frame of the visualization.
     * This should be called in the main application loop.
     * @return bool True if the visualization should continue, false if it
     * should close.
     */
    virtual bool renderFrame() = 0;

    /**
     * @brief Set a status message to be displayed.
     * @param status Status string (e.g., "Training...", "Paused").
     */
    virtual void setStatus(const std::string& status) = 0;

    /**
     * @brief Show/Hide the control panel.
     * @param show Boolean flag.
     */
    virtual void showControlPanel(bool show) = 0;

    /**
     * @brief Get the last action requested by the user from the control panel.
     * @return ControlAction
     */
    [[nodiscard]] virtual ControlAction getControlAction() const = 0;

    /**
     * @brief Check if the user has requested to stop training via the UI.
     * @return true If training should stop.
     */
    virtual bool requestedStop() const = 0;
};

/**
 * @brief Factory function to create a concrete Visualizer instance using ImGui.
 * @return std::unique_ptr<IVisualizer>
 */
std::unique_ptr<IVisualizer> createMLVisualizer();

} // namespace hahaha::display

#endif // HAHAHA_DISPLAY_VISUALIZER_H
