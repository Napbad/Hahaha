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

#include <Res.h>
#include <ds/String.h>
#include <ds/Vector.h>
#include <Error.h>
#include <fstream>
#include <dataset/Dataset.h>
#include <dataset/Sample.h>
#include <sstream>

namespace hahaha::ml
{

using namespace hahaha::common::ds;
/**
 * CSV dataset error class
 */
class CSVDatasetError final : public Error
{
  public:
    explicit CSVDatasetError(ds::String message, ds::String location = ds::String("CSVDataset"))
        : message_(std::move(message)), location_(std::move(location))
    {
    }

    [[nodiscard]] ds::String typeName() const override
    {
        return ds::String("CSVDatasetError");
    }
    [[nodiscard]] ds::String message() const override
    {
        return message_;
    }
    [[nodiscard]] ds::String location() const override
    {
        return location_;
    }
    [[nodiscard]] ds::String toString() const override
    {
        return typeName() + ds::String(": ") + message() + ds::String(" at ") + location();
    }

  private:
    ds::String message_;
    ds::String location_;
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
template <typename T> class CSVDataset : public Dataset<T>
{
  public:
    /**
     * Create a new CSV dataset
     */
    CSVDataset(ds::String filepath,
               const ds::Vector<sizeT>& featureCols,
               const ds::Vector<sizeT>& labelCols,
               const bool hasHeader = true,
               const char delimiter = ',',
               ds::String description = ds::String())
        : filepath_(std::move(filepath)), featureCols_(featureCols.begin(), featureCols.end()),
          labelCols_(labelCols.begin(), labelCols.end()), hasHeader_(hasHeader), delimiter_(delimiter),
          description_(std::move(description))
    {
    }

    ~CSVDataset() override = default;

    /**
     * Load the dataset from file
     */
    Res<void, DatasetError> load() override
    {
        SetRetT(void, DatasetError);

        std::ifstream file(filepath_.data());
        if (!file.is_open())
        {
            Err(DatasetError(ds::String("Failed to open file: ") + filepath_));
        }

        // Skip header if needed
        std::string line;
        if (hasHeader_ && std::getline(file, line))
        {
            parseHeader(line);
        }

        // Read data
        size_t row = 0;
        while (std::getline(file, line))
        {
            auto sampleRes = parseLine(line, row);
            if (sampleRes.isErr())
            {
                Err(sampleRes.unwrapErr());
            }
            samples_.pushBack(sampleRes.unwrap());
            row++;
        }

        Ok();
    }

    /**
     * Get a sample by index
     */
    [[nodiscard]] Res<Sample<T>, DatasetError> get(size_t index) const override
    {
        SetRetT(Sample<T>, DatasetError);

        if (index >= samples_.size())
        {
            Err(DatasetError(ds::String("Index out of bounds")));
        }
        Ok(samples_[index]);
    }

    /**
     * Get the size of the dataset
     */
    [[nodiscard]] size_t size() const override
    {
        return samples_.size();
    }

    /**
     * Get the feature dimension
     */
    [[nodiscard]] size_t featureDim() const override
    {
        return featureCols_.size();
    }

    /**
     * Get the label dimension
     */
    [[nodiscard]] size_t labelDim() const override
    {
        return labelCols_.size();
    }

    /**
     * Get feature names if available
     */
    [[nodiscard]] ds::Vector<ds::String> featureNames() const override
    {
        if (!hasHeader_)
        {
            return {};
        }

        ds::Vector<ds::String> names;
        for (size_t i : featureCols_)
        {
            if (i < columnNames_.size())
            {
                names.pushBack(columnNames_[i]);
            }
        }
        return names;
    }

    /**
     * Get label names if available
     */
    [[nodiscard]] ds::Vector<ds::String> labelNames() const override
    {
        if (!hasHeader_)
        {
            return {};
        }

        ds::Vector<ds::String> names;
        for (size_t i : labelCols_)
        {
            if (i < columnNames_.size())
            {
                names.pushBack(columnNames_[i]);
            }
        }
        return names;
    }

    /**
     * Get dataset description if available
     */
    [[nodiscard]] ds::String description() const override
    {
        return description_;
    }

  private:
    ds::String filepath_;
    ds::Vector<sizeT> featureCols_;
    ds::Vector<sizeT> labelCols_;
    bool hasHeader_;
    char delimiter_;
    ds::String description_;
    ds::Vector<ds::String> columnNames_;
    ds::Vector<Sample<T>> samples_;

    /**
     * Parse a header line
     */
    void parseHeader(const std::string& line)
    {
        std::istringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter_))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);

            columnNames_.pushBack(ds::String(token));
        }
    }

    /**
     * Parse a data line into a Sample
     */
    Res<Sample<T>, DatasetError> parseLine(const std::string& line, size_t row)
    {
        SetRetT(Sample<T>, DatasetError);

        std::istringstream ss(line);
        std::string token;
        ds::Vector<T> values;
        size_t col = 0;

        while (std::getline(ss, token, delimiter_))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);

            // Parse value
            try
            {
                if constexpr (std::is_same_v<T, f32>)
                {
                    values.pushBack(std::stof(token));
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    values.pushBack(std::stod(token));
                }
                else if constexpr (std::is_same_v<T, int>)
                {
                    values.pushBack(std::stoi(token));
                }
                else
                {
                    Err(DatasetError(ds::String("Unsupported data type")));
                }
            }
            catch (const std::exception& e)
            {
                std::ostringstream errMsg;
                errMsg << "Failed to parse value at row " << row << ", column " << col;
                Err(DatasetError(ds::String(errMsg.str())));
            }
            col++;
        }

        // Check if we have enough columns
        if (col < std::max(*std::max_element(featureCols_.begin(), featureCols_.end()),
                           *std::max_element(labelCols_.begin(), labelCols_.end())))
        {
            std::ostringstream errMsg;
            errMsg << "Not enough columns at row " << row;
            Err(DatasetError(ds::String(errMsg.str())));
        }

        // Extract features and labels
        ds::Vector<T> features;
        ds::Vector<T> labels;

        for (size_t i : featureCols_)
        {
            features.pushBack(values[i]);
        }
        for (size_t i : labelCols_)
        {
            labels.pushBack(values[i]);
        }

        // Create tensors
        auto featureTensor = Tensor<T>::fromVector(features);
        auto labelTensor = Tensor<T>::fromVector(labels);

        Ok(Sample<T>(std::move(featureTensor), std::move(labelTensor)));
    }
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_CSVDATASET_H
