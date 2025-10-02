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
// Created by Napbad on 7/14/25.
//

#ifndef MODEL_H
#define MODEL_H

#include "common/ds/Vec.h"
#include <string>

namespace hahaha {

    using namespace hahaha::common;
    class Model {
    public:
        virtual ~Model() = default;

        // Train the model with given features and labels
        virtual void train(const ds::Vec<ds::Vec<float>>& features, const ds::Vec<float>& labels) = 0;

        // Make a prediction given input features
        [[nodiscard]] virtual float predict(const ds::Vec<float>& features) const = 0;

        // Save the model to a file
        [[nodiscard]] virtual bool save(const std::string& filepath) const = 0;

        // Load the model from a file
        virtual bool load(const std::string& filepath) = 0;

        // Get the number of parameters in the model
        [[nodiscard]] virtual size_t parameterCount() const = 0;

        // Get a name describing the model type
        [[nodiscard]] virtual std::string modelName() const = 0;
    };
} // namespace hahaha

#endif // MODEL_H
