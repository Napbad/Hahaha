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
// jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "Tensor.h"
#include "common/definitions.h"

using hahaha::Tensor;
using namespace hahaha::common;

void nn_train_example() {

    std::mt19937 engine;

    std::random_device randomDevice;
    std::seed_seq seed{
        randomDevice(),
        static_cast<unsigned int>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    engine.seed(seed);

    std::uniform_real_distribution<float> dist(0, 100);

    constexpr int DataSize = 100;
    constexpr int TrainLoop = 10;

    std::vector<f32> x;
    std::vector<f32> y;
    x.reserve(DataSize);
    y.reserve(DataSize);

    // y = 2 * x
    for (int i = 0; i < DataSize; ++i) {
        auto val = dist(engine);
        x[i] = val;
        y[i] = val * 2;
    }

    Tensor<f32> xTensor = Tensor<f32>::buildFromVector(x);
    Tensor<f32> yTensor = Tensor<f32>::buildFromVector(y);
    Tensor<f32> w(1);
    // y = w * x
    Tensor<f32> loss(1);

    for (int i = 0; i < TrainLoop; ++i) {
        loss.clearGrad();
        loss.clear();
        for (int i = 0; i < DataSize; ++i) {
            loss = loss + (yTensor - w * xTensor).matmul(yTensor - w * xTensor);
        }
        loss.backward();
    }
}