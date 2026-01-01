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

#include <algorithm>
#include <map>

#include "display/Visualizer.h"
#include "imgui.h"

namespace hahaha::display {

class MLVisualizer : public IVisualizer {
  public:
    void init(const std::string& title, int width, int height) override {
        title_ = title;
        width_ = width;
        height_ = height;
    }

    void clearModel() override {
        layers_.clear();
    }

    void addLayer(const LayerInfo& info) override {
        layers_.push_back(info);
    }

    void recordMetrics(int epoch, float loss, float accuracy) override {
        epochs_.push_back(static_cast<float>(epoch));
        lossHistory_.push_back(loss);
        accHistory_.push_back(accuracy);

        if (lossHistory_.size() > 100) {
            lossHistory_.erase(lossHistory_.begin());
            accHistory_.erase(accHistory_.begin());
            epochs_.erase(epochs_.begin());
        }
    }

    bool renderFrame() override {
        renderControlPanel();
        renderGraph();
        renderMetrics();
        return true;
    }

    void setStatus(const std::string& status) override {
        status_ = status;
    }

    void showControlPanel(bool show) override {
        show_control_ = show;
    }

    [[nodiscard]] ControlAction getControlAction() const override {
        return lastAction_;
    }

    bool requestedStop() const override {
        return lastAction_ == ControlAction::Stop;
    }

    void visualizeTensor(const std::string& name,
                         const math::TensorWrapper<float>& data) override {
        // Simple implementation: just store name for now,
        // real implementation would convert tensor to texture
        activeTensors_[name] = &data;
    }

  private:
    void renderControlPanel() {
        if (!show_control_)
            return;

        ImGui::Begin("Training Control");
        ImGui::Text("Status: %s", status_.c_str());

        lastAction_ = ControlAction::None;
        if (ImGui::Button("Start"))
            lastAction_ = ControlAction::Start;
        ImGui::SameLine();
        if (ImGui::Button("Pause"))
            lastAction_ = ControlAction::Pause;
        ImGui::SameLine();
        if (ImGui::Button("Stop"))
            lastAction_ = ControlAction::Stop;
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            lastAction_ = ControlAction::Reset;

        ImGui::End();
    }

    void renderGraph() {
        ImGui::Begin("Model Architecture");

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 cursorPos = ImGui::GetCursorScreenPos();

        float nodeWidth = 120.0f;
        float nodeHeight = 50.0f;
        float spacingY = 40.0f;

        ImVec2 current_pos = {cursorPos.x + 50, cursorPos.y + 20};

        for (size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];

            // Draw node box
            ImVec2 p_min = current_pos;
            ImVec2 p_max = {current_pos.x + nodeWidth,
                            current_pos.y + nodeHeight};
            drawList->AddRectFilled(
                p_min, p_max, IM_COL32(60, 60, 70, 255), 5.0f);
            drawList->AddRect(p_min, p_max, IM_COL32(200, 200, 200, 255), 5.0f);

            // Draw text
            std::string label = layer.name + "\n("
                + std::to_string(layer.inputSize) + "->"
                + std::to_string(layer.outputSize) + ")";
            ImGui::SetCursorScreenPos({current_pos.x + 5, current_pos.y + 5});
            ImGui::Text("%s", label.c_str());

            // Draw connection to next
            if (i < layers_.size() - 1) {
                ImVec2 start = {current_pos.x + nodeWidth / 2,
                                current_pos.y + nodeHeight};
                ImVec2 end = {current_pos.x + nodeWidth / 2,
                              current_pos.y + nodeHeight + spacingY};
                drawList->AddLine(
                    start, end, IM_COL32(255, 255, 255, 255), 2.0f);
            }

            current_pos.y += nodeHeight + spacingY;
        }

        ImGui::End();
    }

    void renderMetrics() {
        ImGui::Begin("Training Metrics");

        if (!lossHistory_.empty()) {
            ImGui::PlotLines("Loss",
                             lossHistory_.data(),
                             static_cast<int>(lossHistory_.size()),
                             0,
                             nullptr,
                             0.0f,
                             FLT_MAX,
                             ImVec2(0, 80));
            ImGui::PlotLines("Accuracy",
                             accHistory_.data(),
                             static_cast<int>(accHistory_.size()),
                             0,
                             nullptr,
                             0.0f,
                             1.0f,
                             ImVec2(0, 80));
        } else {
            ImGui::Text("No metrics recorded yet.");
        }

        ImGui::End();
    }

    std::string title_;
    int width_, height_;
    std::string status_ = "Idle";
    bool show_control_ = true;
    ControlAction lastAction_ = ControlAction::None;

    std::vector<LayerInfo> layers_;
    std::vector<float> lossHistory_;
    std::vector<float> accHistory_;
    std::vector<float> epochs_;
    std::map<std::string, const math::TensorWrapper<float>*> activeTensors_;
};

std::unique_ptr<IVisualizer> createMLVisualizer() {
    return std::make_unique<MLVisualizer>();
}

} // namespace hahaha::display
