#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <Eigen/Dense>

namespace mnist {

/**
 * @brief Class for loading and processing MNIST dataset
 * 
 * This class handles loading MNIST data from the standard MNIST format,
 * normalizing the data, and providing batches for training and testing.
 */
class Dataset {
public:
    static constexpr int kImageSize = 28;
    static constexpr int kInputSize = kImageSize * kImageSize;
    static constexpr int kNumClasses = 10;
    
    using Image = Eigen::VectorXf;
    using Label = Eigen::VectorXf;
    using Batch = std::pair<std::vector<Image>, std::vector<Label>>;
    
    /**
     * @brief Construct a Dataset object
     * 
     * @param data_dir Directory containing MNIST data files
     */
    explicit Dataset(const std::string& data_dir);

    /**
     * @brief Load MNIST dataset from files
     * 
     * @param train_images_path Path to training images file
     * @param train_labels_path Path to training labels file
     * @param test_images_path Path to test images file
     * @param test_labels_path Path to test labels file
     * @return true if loading was successful
     */
    bool load(const std::string& train_images_path,
              const std::string& train_labels_path,
              const std::string& test_images_path,
              const std::string& test_labels_path);

    /**
     * @brief Get the next batch of training data
     * 
     * @param batch_size Number of samples in the batch
     * @return Batch of images and one-hot encoded labels
     */
    Batch getTrainBatch(size_t batch_size);

    /**
     * @brief Get the test data
     * 
     * @return Batch of all test images and one-hot encoded labels
     */
    Batch getTestData() const;

    /**
     * @brief Get number of training samples
     * 
     * @return Number of training samples
     */
    size_t getTrainSize() const { return m_train_images.size(); }

    /**
     * @brief Get number of test samples
     * 
     * @return Number of test samples
     */
    size_t getTestSize() const { return m_test_images.size(); }

private:
    // Training data
    std::vector<Image> m_train_images;
    std::vector<Label> m_train_labels;
    
    // Test data
    std::vector<Image> m_test_images;
    std::vector<Label> m_test_labels;
    
    // Current position in training data for batch generation
    size_t m_current_pos = 0;
    
    /**
     * @brief Convert raw image data to normalized vector
     * 
     * @param raw_data Raw image bytes
     * @return Normalized image vector
     */
    Image preprocessImage(const std::vector<unsigned char>& raw_data) const;
    
    /**
     * @brief Convert label to one-hot encoded vector
     * 
     * @param label Raw label (0-9)
     * @return One-hot encoded label
     */
    Label oneHotEncode(unsigned char label) const;

    /**
     * @brief Read MNIST image file
     * 
     * @param filename Path to MNIST image file
     * @param images Output vector to store processed images
     * @return true if successful
     */
    bool readImageFile(const std::string& filename, std::vector<Image>& images);

    /**
     * @brief Read MNIST label file
     * 
     * @param filename Path to MNIST label file
     * @param labels Output vector to store processed labels
     * @return true if successful
     */
    bool readLabelFile(const std::string& filename, std::vector<Label>& labels);
};

} // namespace mnist
