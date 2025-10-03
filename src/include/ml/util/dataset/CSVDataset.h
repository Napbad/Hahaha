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

#ifndef HIAHIAHIA_CSVDATASET_H
#define HIAHIAHIA_CSVDATASET_H

#include <common/Res.h>
#include <common/ds/Str.h>
#include <common/ds/Vec.h>
#include <common/error.h>
#include <fstream>
#include <ml/util/dataset/Dataset.h>
#include <ml/util/dataset/Sample.h>
#include <sstream>

namespace hahaha::ml {

using namespace hahaha::common::ds;
    /**
     * CSV dataset error class
     */
    class CSVDatasetError final : public error {
    public:
        explicit CSVDatasetError(Str message, Str location = Str("CSVDataset"))
            : _message(std::move(message)), _location(std::move(location)) {}

        [[nodiscard]] Str typeName() const override {
            return Str("CSVDatasetError");
        }
        [[nodiscard]] Str message() const override {
            return _message;
        }
        [[nodiscard]] Str location() const override {
            return _location;
        }
        [[nodiscard]] Str toString() const override {
            return typeName() + Str(": ") + message() + Str(" at ") + location();
        }

    private:
        Str _message;
        Str _location;
    };

    /**
     * CSV dataset class
     *
     * Loads data from CSV files with support for:
     * - Header row
     * - Feature columns
     * - Label columns
     * - Custom delimiter
     */
    template <typename T>
    class CSVDataset : public Dataset<T> {
    public:
        /**
         * Create a new CSV dataset
         */
        CSVDataset(Str filepath, const Vec<sizeT>& featureCols, const Vec<sizeT>& labelCols,
            const bool hasHeader = true, const char delimiter = ',', Str description = Str())
            : _filepath(std::move(filepath)), _featureCols(featureCols.begin(), featureCols.end()),
        _labelCols(labelCols.begin(), labelCols.end()), _hasHeader(hasHeader),
              _delimiter(delimiter), _description(std::move(description)) {}

        ~CSVDataset() override = default;

        /**
         * Load the dataset from file
         */
        Res<void, DatasetError> load() override {
            SetRetT(void, DatasetError);

            std::ifstream file(_filepath.data());
            if (!file.is_open()) {
                Err(DatasetError(Str("Failed to open file: ") + _filepath));
            }

            // Skip header if needed
            std::string line;
            if (_hasHeader && std::getline(file, line)) {
                parseHeader(line);
            }

            // Read data
            size_t row = 0;
            while (std::getline(file, line)) {
                auto sampleRes = parseLine(line, row);
                if (sampleRes.isErr()) {
                    Err(sampleRes.unwrapErr());
                }
                _samples.push_back(sampleRes.unwrap());
                row++;
            }

            Ok();
        }

        /**
         * Get a sample by index
         */
        [[nodiscard]] Res<Sample<T>, DatasetError> get(size_t index) const override {
            SetRetT(Sample<T>, DatasetError);

            if (index >= _samples.size()) {
                Err(DatasetError(Str("Index out of bounds")));
            }
            Ok(_samples[index]);
        }

        /**
         * Get the size of the dataset
         */
        [[nodiscard]] size_t size() const override {
            return _samples.size();
        }

        /**
         * Get the feature dimension
         */
        [[nodiscard]] size_t featureDim() const override {
            return _featureCols.size();
        }

        /**
         * Get the label dimension
         */
        [[nodiscard]] size_t labelDim() const override {
            return _labelCols.size();
        }

        /**
         * Get feature names if available
         */
        [[nodiscard]] Vec<Str> featureNames() const override {
            if (!_hasHeader) {
                return {};
            }

            Vec<Str> names;
            for (size_t i : _featureCols) {
                if (i < _columnNames.size()) {
                    names.push_back(_columnNames[i]);
                }
            }
            return names;
        }

        /**
         * Get label names if available
         */
        [[nodiscard]] Vec<Str> labelNames() const override {
            if (!_hasHeader) {
                return {};
            }

            Vec<Str> names;
            for (size_t i : _labelCols) {
                if (i < _columnNames.size()) {
                    names.push_back(_columnNames[i]);
                }
            }
            return names;
        }

        /**
         * Get dataset description if available
         */
        [[nodiscard]] Str description() const override {
            return _description;
        }

    private:
        Str _filepath;
        Vec<sizeT> _featureCols;
        Vec<sizeT> _labelCols;
        bool _hasHeader;
        char _delimiter;
        Str _description;
        Vec<Str> _columnNames;
        Vec<Sample<T>> _samples;

        /**
         * Parse a header line
         */
        void parseHeader(const std::string& line) {
            std::istringstream ss(line);
            std::string token;

            while (std::getline(ss, token, _delimiter)) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);

                _columnNames.push_back(Str(token));
            }
        }

        /**
         * Parse a data line into a Sample
         */
        Res<Sample<T>, DatasetError> parseLine(const std::string& line, size_t row) {
            SetRetT(Sample<T>, DatasetError);

            std::istringstream ss(line);
            std::string token;
            Vec<T> values;
            size_t col = 0;

            while (std::getline(ss, token, _delimiter)) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);

                // Parse value
                try {
                    if constexpr (std::is_same_v<T, f32>) {
                        values.push_back(std::stof(token));
                    } else if constexpr (std::is_same_v<T, double>) {
                        values.push_back(std::stod(token));
                    } else if constexpr (std::is_same_v<T, int>) {
                        values.push_back(std::stoi(token));
                    } else {
                        Err(DatasetError(Str("Unsupported data type")));
                    }
                } catch (const std::exception& e) {
                    std::ostringstream errMsg;
                    errMsg << "Failed to parse value at row " << row << ", column " << col;
                    Err(DatasetError(Str(errMsg.str())));
                }
                col++;
            }

            // Check if we have enough columns
            if (col < std::max(*std::max_element(_featureCols.begin(), _featureCols.end()),
                    *std::max_element(_labelCols.begin(), _labelCols.end()))) {
                std::ostringstream errMsg;
                errMsg << "Not enough columns at row " << row;
                Err(DatasetError(Str(errMsg.str())));
            }

            // Extract features and labels
            Vec<T> features;
            Vec<T> labels;

            for (size_t i : _featureCols) {
                features.push_back(values[i]);
            }
            for (size_t i : _labelCols) {
                labels.push_back(values[i]);
            }

            // Create tensors
            auto featureTensor = Tensor<T>::fromVector(features);
            auto labelTensor   = Tensor<T>::fromVector(labels);

            Ok(Sample<T>(std::move(featureTensor), std::move(labelTensor)));
        }
    };

} // namespace hahaha::ml

#endif // HIAHIAHIA_CSVDATASET_H
