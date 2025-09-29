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

#ifndef HIAHIAHIA_DATASETDOWNLOADER_H
#define HIAHIAHIA_DATASETDOWNLOADER_H

#include <common/Error.h>
#include <common/Res.h>
#include <common/ds/str.h>

namespace hahaha::ml {

  /**
   * Dataset downloader error class
   */
  class DatasetDownloaderError final : public Error {
public:
    explicit DatasetDownloaderError(const ds::Str &message, const ds::Str &location = ds::Str("DatasetDownloader")) :
        _message(message), _location(location) {}

    explicit DatasetDownloaderError(const char *message, const ds::Str &location = ds::Str("DatasetDownloader")) :
        _message(ds::Str(message)), _location(location) {}

    [[nodiscard]] ds::Str typeName() const override { return ds::Str("DatasetDownloaderError"); }
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
   * Dataset downloader class
   *
   * Provides functionality to download datasets from various sources:
   * - Direct URL
   * - UCI Machine Learning Repository
   * - Kaggle
   */
  class DatasetDownloader {
public:
    /**
     * Download a dataset from a direct URL
     */
    static Res<void, DatasetDownloaderError> downloadFromUrl(const ds::Str &url, const ds::Str &outputPath,
                                                             bool showProgress = false);

    /**
     * Download a dataset from UCI Machine Learning Repository
     */
    static Res<void, DatasetDownloaderError> downloadFromUCI(const ds::Str &datasetName, const ds::Str &outputPath);

    /**
     * Download a dataset from Kaggle
     */
    static Res<void, DatasetDownloaderError> downloadFromKaggle(const ds::Str &datasetName, const ds::Str &outputPath,
                                                                const ds::Str &apiToken);
  };

} // namespace hahaha::ml

#endif // HIAHIAHIA_DATASETDOWNLOADER_H
