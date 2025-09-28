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

#ifndef HIAHIAHIA_DATASET_H
#define HIAHIAHIA_DATASET_H

#include <common/Error.h>
#include <common/Res.h>
#include <common/ds/Vec.h>
#include <common/ds/str.h>
#include <ml/common/Tensor.h>

#include "Sample.h"

namespace hahaha {
namespace ml {

/**
 * Dataset error class
 */
class DatasetError : public Error {
public:
  explicit DatasetError(ds::Str message, ds::Str location = ds::Str("dataset"))
    : _message(std::move(message)), _location(std::move(location)) {}

  [[nodiscard]] ds::Str typeName() const override { return ds::Str("DatasetError"); }
  [[nodiscard]] ds::Str message() const override { return _message; }
  [[nodiscard]] ds::Str location() const override { return _location; }
  [[nodiscard]] ds::Str toString() const override { 
    return typeName() + ds::Str(": ") + message() + ds::Str(" at ") + location();
  }

private:
  ds::Str _message;
  ds::Str _location;
};

/**
 * Dataset interface
 * 
 * Base class for all dataset implementations
 */
template<typename T>
class Dataset {
public:
  virtual ~Dataset() = default;

  /**
   * Load the dataset
   */
  virtual Res<void, DatasetError> load() = 0;

  /**
   * Get the size of the dataset
   */
  [[nodiscard]] virtual size_t size() const = 0;

  /**
   * Get a sample by index
   */
  [[nodiscard]] virtual Res<Sample<T>, DatasetError> get(size_t index) const = 0;

  /**
   * Get the feature dimension
   */
  [[nodiscard]] virtual size_t featureDim() const = 0;

  /**
   * Get the label dimension
   */
  [[nodiscard]] virtual size_t labelDim() const = 0;

  /**
   * Get feature names if available
   */
  [[nodiscard]] virtual ds::Vec<ds::Str> featureNames() const {
    return ds::Vec<ds::Str>();
  }

  /**
   * Get label names if available
   */
  [[nodiscard]] virtual ds::Vec<ds::Str> labelNames() const {
    return ds::Vec<ds::Str>();
  }

  /**
   * Get dataset description if available
   */
  [[nodiscard]] virtual ds::Str description() const {
    return ds::Str();
  }
};

} // namespace ml
} // namespace hahaha

#endif // HIAHIAHIA_DATASET_H
