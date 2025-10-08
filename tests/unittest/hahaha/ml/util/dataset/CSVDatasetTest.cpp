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

//
// Created by root on 10/2/25.
//

#include <gtest/gtest.h>
#include <fstream>

#include <ds/Vector.h>
#include <dataset/CSVDataset.h>

using namespace hahaha::ml;
using namespace hahaha;
// CSVDataset tests
class CSVDatasetTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Create a temporary CSV file for testing
        csvContent = "feature1,feature2,label\n"
                     "1.0,2.0,0\n"
                     "3.0,4.0,1\n"
                     "5.0,6.0,0\n";

        std::ofstream file("test_data.csv");
        file << csvContent;
        file.close();

        // Create CSV without header
        csvContentNoHeader = "1.0,2.0,0\n"
                             "3.0,4.0,1\n"
                             "5.0,6.0,0\n";

        std::ofstream fileNoHeader("test_data_no_header.csv");
        fileNoHeader << csvContentNoHeader;
        fileNoHeader.close();
    }

    void TearDown() override
    {
        // Clean up temporary files
        std::remove("test_data.csv");
        std::remove("test_data_no_header.csv");
        std::remove("nonexistent.csv");
    }

    std::string csvContent;
    std::string csvContentNoHeader;
};

TEST_F(CSVDatasetTest, Constructor)
{
    const ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    const ds::Vector<hahaha::sizeT> labelCols = {2};

    const CSVDataset<f32> dataset(ds::String("test_data.csv"), featureCols, labelCols, true, ',', ds::String("Test dataset"));

    EXPECT_EQ(dataset.featureDim(), 2);
    EXPECT_EQ(dataset.labelDim(), 1);
    EXPECT_EQ(dataset.description(), ds::String("Test dataset"));
}

TEST_F(CSVDatasetTest, LoadSuccess)
{
    const ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    const ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data.csv"), featureCols, labelCols);
    const auto result = dataset.load();

    EXPECT_TRUE(result.isOk());
    EXPECT_EQ(dataset.size(), 3);
}

TEST_F(CSVDatasetTest, LoadFailure)
{
    const ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    const ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("nonexistent.csv"), featureCols, labelCols);
    const auto result = dataset.load();

    EXPECT_TRUE(result.isErr());
}

TEST_F(CSVDatasetTest, GetData)
{
    ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data.csv"), featureCols, labelCols);
    auto loadResult = dataset.load();
    ASSERT_TRUE(loadResult.isOk());

    // Test valid index
    auto getResult = dataset.get(0);
    EXPECT_TRUE(getResult.isOk());

    auto sample = getResult.unwrap();
    EXPECT_EQ(sample.features().size(), 2);
    EXPECT_EQ(sample.labels().size(), 1);

    // Test out of bounds
    auto outOfBoundsResult = dataset.get(10);
    EXPECT_TRUE(outOfBoundsResult.isErr());
}

TEST_F(CSVDatasetTest, FeatureAndLabelNames)
{
    const ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    const ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data.csv"), featureCols, labelCols, true);
    const auto loadResult = dataset.load();
    ASSERT_TRUE(loadResult.isOk());

    auto featureNames = dataset.featureNames();
    auto labelNames = dataset.labelNames();

    EXPECT_EQ(featureNames.size(), 2);
    EXPECT_EQ(labelNames.size(), 1);
    EXPECT_EQ(featureNames[0], ds::String("feature1"));
    EXPECT_EQ(featureNames[1], ds::String("feature2"));
    EXPECT_EQ(labelNames[0], ds::String("label"));
}

TEST_F(CSVDatasetTest, NoHeader)
{
    const ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    const ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data_no_header.csv"), featureCols, labelCols, false);
    const auto loadResult = dataset.load();
    ASSERT_TRUE(loadResult.isOk());

    const auto featureNames = dataset.featureNames();
    const auto labelNames = dataset.labelNames();

    EXPECT_TRUE(featureNames.empty());
    EXPECT_TRUE(labelNames.empty());
}

TEST_F(CSVDatasetTest, DifferentDelimiter)
{
    // Create CSV with semicolon delimiter
    std::string csvSemicolon = "feature1;feature2;label\n"
                               "1.0;2.0;0\n"
                               "3.0;4.0;1\n";

    std::ofstream file("test_data_semicolon.csv");
    file << csvSemicolon;
    file.close();

    ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data_semicolon.csv"), featureCols, labelCols, true, ';');
    auto loadResult = dataset.load();
    ASSERT_TRUE(loadResult.isOk());
    EXPECT_EQ(dataset.size(), 2);

    // Cleanup
    std::remove("test_data_semicolon.csv");
}

TEST_F(CSVDatasetTest, ParseError)
{
    // Create CSV with invalid data
    std::string csvInvalid = "feature1,feature2,label\n"
                             "1.0,2.0,0\n"
                             "invalid,4.0,1\n";

    std::ofstream file("test_data_invalid.csv");
    file << csvInvalid;
    file.close();

    ds::Vector<hahaha::sizeT> featureCols = {0, 1};
    ds::Vector<hahaha::sizeT> labelCols = {2};

    CSVDataset<f32> dataset(ds::String("test_data_invalid.csv"), featureCols, labelCols);
    auto loadResult = dataset.load();

    // Should fail due to invalid data
    EXPECT_TRUE(loadResult.isErr());

    // Cleanup
    std::remove("test_data_invalid.csv");
}

TEST(DatasetErrorTest, ErrorHandling)
{
    const DatasetError error(ds::String("Test error"), ds::String("test_location"));

    EXPECT_EQ(error.typeName(), ds::String("DatasetError"));
    EXPECT_EQ(error.message(), ds::String("Test error"));
    EXPECT_EQ(error.location(), ds::String("test_location"));
    EXPECT_EQ(error.toString(), ds::String("DatasetError: Test error at test_location"));
}

TEST(CSVDatasetErrorTest, ErrorHandling)
{
    const CSVDatasetError error(ds::String("CSV Test error"), ds::String("csv_test_location"));

    EXPECT_EQ(error.typeName(), ds::String("CSVDatasetError"));
    EXPECT_EQ(error.message(), ds::String("CSV Test error"));
    EXPECT_EQ(error.location(), ds::String("csv_test_location"));
    EXPECT_EQ(error.toString(), ds::String("CSVDatasetError: CSV Test error at csv_test_location"));
}
