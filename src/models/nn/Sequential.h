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
// Created by Napbad on 2025/10/9.
//

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "compute/Variable.h"
#include "ds/Vector.h"
#include "layers/Layer.h"

namespace hahaha::ml {

template<typename T>
class Sequential final : public Layer<T> {
public:
    Sequential() = default;
    Sequential(std::initializer_list<Layer<T>*> layers) : layers_(layers) {}

    void add(Layer<T>* layer) {
        layers_.pushBack(layer);
    }

    Variable<T> forward(const Variable<T>& input) override {
        Variable<T> current_output = input;
        for (auto& layer : layers_) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    ds::Vector<Variable<T>*> parameters() override {
        ds::Vector<Variable<T>*> params;
        for (auto& layer : layers_) {
            for (auto layer_params = layer->parameters(); auto* p : layer_params) {
                params.pushBack(p);
            }
        }
        return params;
    }

private:
    ds::Vector<Layer<T>*> layers_;
};

} // namespace hahaha::ml

#endif //SEQUENTIAL_H
