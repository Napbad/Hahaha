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

#include <ml/model/linear_regression/LinearRegression.h>

namespace hahaha {

  using namespace hahaha::common;

float LinearRegression::predict(const ds::Vec<float>& features) const {
  float result = _bias;
  for (size_t i = 0; i < features.size(); ++i) {
    result += features[i] * _weights[i];
  }
  return result;
}

void LinearRegression::train(const ds::Vec<ds::Vec<float>>& features, const ds::Vec<float>& labels) {
  // Initialize weights if not already done
  if (_weights.size() == 0) {
    // Create weights vector with zeros
    for (size_t i = 0; i < features[0].size(); ++i) {
      _weights.push_back(0.0f);
    }
    _bias = 0.0f;
  }

  // Simple gradient descent
  const float learningRate = 0.01f;
  const int numEpochs = 100;

  for (int epoch = 0; epoch < numEpochs; ++epoch) {
    for (size_t i = 0; i < features.size(); ++i) {
      // Forward pass
      float prediction = predict(features[i]);
      float error = prediction - labels[i];

      // Update weights
      for (size_t j = 0; j < _weights.size(); ++j) {
        _weights[j] -= learningRate * error * features[i][j];
      }

      // Update bias
      _bias -= learningRate * error;
    }
  }
}

} // namespace hahaha