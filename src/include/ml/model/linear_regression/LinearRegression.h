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

#include <ml/model/Model.h>
#include <common/ds/Vec.h>

namespace hiahiahia {

class LinearRegression : public Model {
public:
  LinearRegression() = default;

  void train(const ds::Vec<ds::Vec<float>>& features, const ds::Vec<float>& labels) override;
  float predict(const ds::Vec<float>& features) const override;

private:
  ds::Vec<float> _weights;
  float _bias{0.0f};
};

} // namespace hiahiahia

#endif // HIAHIAHIA_LINEARREGRESSION_H
