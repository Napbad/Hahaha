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

#ifndef HIAHIAHIA_LINEARREGRESSION_H
#define HIAHIAHIA_LINEARREGRESSION_H

#include "ml/common/Tensor.h"
#include <common/ds/Vec.h>
#include <ml/model/Model.h>

namespace hahaha {

    class LinearRegression : public Model {
    public:
        LinearRegression() = default;

        LinearRegression(const f32 learningRate, const sizeT epochs) : _learningRate(learningRate), _epochs(epochs) {}

        [[nodiscard]] Res<void, BaseErr> checkStatus(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels) const;
        Res<ml::TrainStatistics, BaseErr> train(const ml::Tensor<f32>& features, const ml::Tensor<f32>& labels) override;
        [[nodiscard]] f32 predict(const ml::Tensor<f32>& features) const override;
        [[nodiscard]] bool save(const Str& filepath) const override;
        bool load(const Str& path) override;
        [[nodiscard]] sizeT parameterCount() const override;
        [[nodiscard]] Str modelName() const override;

        void setLearningRate(const f32 learningRate) {
            _learningRate = learningRate;
        }
        void setEpochs(const sizeT epochs) {
            _epochs = epochs;
        }
        [[nodiscard]] f32 learningRate() const {
            return _learningRate;
        }

        [[nodiscard]] sizeT epochs() const {
            return _epochs;
        }

    private:
        ml::Tensor<f32> _weights;
        f32 _bias{0.0f};

        f32 _learningRate{0.01f};
        sizeT _epochs{1000};
    };

} // namespace hahaha

#endif // HIAHIAHIA_LINEARREGRESSION_H
