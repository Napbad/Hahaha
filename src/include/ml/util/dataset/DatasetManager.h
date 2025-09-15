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

#ifndef HIAHIAHIA_DATASETMANAGER_H
#define HIAHIAHIA_DATASETMANAGER_H

#include <common/ds/Vec.h>
#include <common/ds/str.h>
#include <common/Res.h>
#include <common/Error.h>
#include <ml/util/dataset/Dataset.h>
#include <ml/util/dataset/CSVDataset.h>
#include <ml/util/dataset/DataLoader.h>
#include <memory>
#include <unordered_map>

namespace hiahiahia::ml {

/**
 * Dataset source type
 */
enum class DatasetSource {
  UCI,      // UCI Machine Learning Repository
  KAGGLE,   // Kaggle
  SKLEARN,  // Scikit-learn built-in datasets
  URL       // Direct URL
};

/**
 * Dataset configuration
 */
struct DatasetConfig {
  ds::Vec<size_t> featureCols;  // Feature column indices
  ds::Vec<size_t> labelCols;    // Label column indices
  bool hasHeader;               // Whether the dataset has a header row
  char delimiter;               // CSV delimiter character
  ds::Str url;                 // Direct URL if source is URL
  ds::Str owner;               // Dataset owner (for Kaggle)
};

/**
 * Dataset manager error class
 */
class DatasetManagerError final : public Error {
public:
  explicit DatasetManagerError(const ds::Str& message, const ds::Str& location = ds::Str("DatasetManager")) :
    _message(message), _location(location) {}

  explicit DatasetManagerError(const char* message, const ds::Str& location = ds::Str("DatasetManager")) :
    _message(ds::Str(message)), _location(location) {}

  [[nodiscard]] ds::Str typeName() const override { return ds::Str("DatasetManagerError"); }
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
 * Dataset manager class
 * 
 * Provides a simple interface to download and load common datasets
 */
class DatasetManager {
public:
  /**
   * Get a dataset by name and source
   */
  static Res<std::shared_ptr<DataLoader<float>>, DatasetManagerError> getDataset(
    const ds::Str& name,
    DatasetSource source = DatasetSource::SKLEARN,
    size_t batchSize = 32,
    bool shuffle = true
  );

  /**
   * Get a dataset with custom configuration
   */
  static Res<std::shared_ptr<DataLoader<float>>, DatasetManagerError> getDatasetWithConfig(
    const ds::Str& name,
    const DatasetConfig& config,
    DatasetSource source = DatasetSource::SKLEARN,
    size_t batchSize = 32,
    bool shuffle = true
  );

private:
  static ds::Str getDatasetPath(const ds::Str& name);
  static ds::Str getDatasetUrl(const ds::Str& name, DatasetSource source, const DatasetConfig& config);

  // Pre-configured datasets
  static const std::unordered_map<ds::Str, DatasetConfig> _datasetConfigs;
};

// Initialize the pre-configured datasets
const std::unordered_map<ds::Str, DatasetConfig> DatasetManager::_datasetConfigs = {
  {ds::Str("boston_house"), DatasetConfig{
    ds::Vec<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // feature columns
    ds::Vec<size_t>{13},                                         // label column
    true,                                                        // has header
    ',',                                                         // delimiter
    ds::Str("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv"),
    ds::Str()                                                    // no owner
  }},
  
  {ds::Str("iris"), DatasetConfig{
    ds::Vec<size_t>{0, 1, 2, 3},  // feature columns
    ds::Vec<size_t>{4},           // label column
    false,                        // no header
    ',',                          // delimiter
    ds::Str(),                    // UCI dataset, URL not needed
    ds::Str()                     // no owner
  }},
  
  {ds::Str("diabetes"), DatasetConfig{
    ds::Vec<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  // feature columns
    ds::Vec<size_t>{10},                             // label column
    false,                                           // no header
    ',',                                            // delimiter
    ds::Str("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/diabetes_data.csv.gz"),
    ds::Str()                                        // no owner
  }}
};

} // namespace hiahiahia::ml

#endif // HIAHIAHIA_DATASETMANAGER_H 