// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
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
// Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <iostream>
#include <ranges>

#include "ml/Tensor.h"
#include "backend/Device.h"

namespace h3::ml {

using backend::Device;

class Module {
public:
    virtual ~Module() = default;

    // Forward pass - to be implemented by subclasses
    virtual Tensor forward(const Tensor& input) = 0;

    // Parameter registration
    void register_parameter(const std::string& name, const Tensor& param) {
        parameters_[name] = param;
    }

    void register_module(const std::string& name, const std::shared_ptr<Module>& module) {
        submodules_[name] = module;
    }

    std::vector<Tensor> parameters() const {
        std::vector<Tensor> params;
        for (const auto& p : parameters_) {
            params.push_back(p.second);
        }
        for (const auto& m : submodules_) {
            auto sub_params = m.second->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    void to(Device device) {
        // Move all parameters to device
        // Simplified implementation
    }

    void train(const bool mode = true) {
        training_ = mode;
        for (const auto& val : submodules_ | std::views::values) {
            val->train(mode);
        }
    }

    void eval() {
        train(false);
    }

protected:
    bool training_ = true;
    std::map<std::string, Tensor> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};

}