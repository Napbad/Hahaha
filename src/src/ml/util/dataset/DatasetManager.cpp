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

#include <ml/util/dataset/DatasetManager.h>
#include <filesystem>

namespace hahaha::ml {

ds::Str DatasetManager::getDatasetPath(const ds::Str& name) {
  return name + ds::Str(".csv");
}

ds::Str DatasetManager::getDatasetUrl(const ds::Str& name, DatasetSource source, const DatasetConfig& config) {
  switch (source) {
    case DatasetSource::UCI:
      return ds::Str("https://archive.ics.uci.edu/ml/machine-learning-databases/") + 
             name + ds::Str("/") + name + ds::Str(".data");
    
    case DatasetSource::SKLEARN:
    case DatasetSource::URL:
      return config.url;
    
    case DatasetSource::KAGGLE:
      return ds::Str("https://www.kaggle.com/api/v1/datasets/download/") + 
             config.owner + ds::Str("/") + name;
    
    default:
      return {};
  }
}

Res<std::shared_ptr<DataLoader<float>>, DatasetManagerError> DatasetManager::getDataset(
  const ds::Str& name,
  DatasetSource source,
  size_t batchSize,
  bool shuffle
) {
  SetRetT(std::shared_ptr<DataLoader<float>>, DatasetManagerError);

  // Find pre-configured dataset
  auto it = _datasetConfigs.find(name);
  if (it == _datasetConfigs.end()) {
    Err(DatasetManagerError(ds::Str("Dataset not found: ") + name));
  }

  return getDatasetWithConfig(name, it->second, source, batchSize, shuffle);
}

Res<std::shared_ptr<DataLoader<float>>, DatasetManagerError> DatasetManager::getDatasetWithConfig(
  const ds::Str& name,
  const DatasetConfig& config,
  DatasetSource source,
  size_t batchSize,
  bool shuffle
) {
  SetRetT(std::shared_ptr<DataLoader<float>>, DatasetManagerError);

  // Get dataset path and URL
  ds::Str path = getDatasetPath(name);
  ds::Str url = getDatasetUrl(name, source, config);

  // Create dataset
  auto dataset = std::make_shared<CSVDataset<float>>(
    path,
    config.featureCols,
    config.labelCols,
    config.hasHeader,
    config.delimiter
  );

  // Load the dataset
  auto loadResult = dataset->load();
  if (loadResult.isErr()) {
    Err(DatasetManagerError(loadResult.unwrapErr().message()));
  }

  // Create and return data loader
  auto loader = std::make_shared<DataLoader<float>>(dataset, batchSize, shuffle);
  Ok(loader);
}

} // namespace hahaha::ml
