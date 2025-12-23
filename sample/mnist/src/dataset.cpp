#include "dataset.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

namespace mnist {

Dataset::Dataset(const std::string& data_dir) {
    // Default MNIST filenames
    const std::string train_images = data_dir + "/train-images-idx3-ubyte";
    const std::string train_labels = data_dir + "/train-labels-idx1-ubyte";
    const std::string test_images = data_dir + "/t10k-images-idx3-ubyte";
    const std::string test_labels = data_dir + "/t10k-labels-idx1-ubyte";
    
    // Load the dataset
    if (!load(train_images, train_labels, test_images, test_labels)) {
        std::cerr << "Failed to load MNIST dataset from " << data_dir << std::endl;
    }
}

bool Dataset::load(const std::string& train_images_path,
                   const std::string& train_labels_path,
                   const std::string& test_images_path,
                   const std::string& test_labels_path) {
    
    // Load training images and labels
    if (!readImageFile(train_images_path, m_train_images) ||
        !readLabelFile(train_labels_path, m_train_labels)) {
        return false;
    }
    
    // Load test images and labels
    if (!readImageFile(test_images_path, m_test_images) ||
        !readLabelFile(test_labels_path, m_test_labels)) {
        return false;
    }
    
    // Verify data consistency
    if (m_train_images.size() != m_train_labels.size() ||
        m_test_images.size() != m_test_labels.size()) {
        std::cerr << "Inconsistent dataset sizes!" << std::endl;
        return false;
    }
    
    std::cout << "MNIST dataset loaded successfully:" << std::endl;
    std::cout << "  Training samples: " << m_train_images.size() << std::endl;
    std::cout << "  Test samples: " << m_test_images.size() << std::endl;
    
    return true;
}

Dataset::Batch Dataset::getTrainBatch(size_t batch_size) {
    Batch batch;
    batch.first.reserve(batch_size);
    batch.second.reserve(batch_size);
    
    // Ensure we don't exceed the number of training samples
    size_t actual_batch_size = std::min(batch_size, m_train_images.size() - m_current_pos);
    
    // If we're at the end of the dataset, shuffle and start over
    if (actual_batch_size < batch_size) {
        // Create indices for all training samples
        std::vector<size_t> indices(m_train_images.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        // Shuffle indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Rearrange data based on shuffled indices
        std::vector<Image> shuffled_images(m_train_images.size());
        std::vector<Label> shuffled_labels(m_train_labels.size());
        
        for (size_t i = 0; i < indices.size(); ++i) {
            shuffled_images[i] = m_train_images[indices[i]];
            shuffled_labels[i] = m_train_labels[indices[i]];
        }
        
        m_train_images = std::move(shuffled_images);
        m_train_labels = std::move(shuffled_labels);
        
        m_current_pos = 0;
        actual_batch_size = std::min(batch_size, m_train_images.size());
    }
    
    // Extract batch
    for (size_t i = 0; i < actual_batch_size; ++i) {
        batch.first.push_back(m_train_images[m_current_pos]);
        batch.second.push_back(m_train_labels[m_current_pos]);
        ++m_current_pos;
    }
    
    return batch;
}

Dataset::Batch Dataset::getTestData() const {
    return { m_test_images, m_test_labels };
}

Dataset::Image Dataset::preprocessImage(const std::vector<unsigned char>& raw_data) const {
    Image image(kInputSize);
    
    // Normalize pixel values to [0, 1]
    for (size_t i = 0; i < raw_data.size(); ++i) {
        image(i) = static_cast<float>(raw_data[i]) / 255.0f;
    }
    
    return image;
}

Dataset::Label Dataset::oneHotEncode(unsigned char label) const {
    Label encoded = Label::Zero(kNumClasses);
    encoded(label) = 1.0f;
    return encoded;
}

bool Dataset::readImageFile(const std::string& filename, std::vector<Image>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Read the magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = __builtin_bswap32(magic); // Convert from big endian
    
    // Check magic number (2051 for image file)
    if (magic != 2051) {
        std::cerr << "Invalid magic number in image file: " << filename << std::endl;
        return false;
    }
    
    // Read number of images, rows, and columns
    uint32_t num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    // Convert from big endian
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    // Verify dimensions
    if (num_rows != kImageSize || num_cols != kImageSize) {
        std::cerr << "Unexpected image dimensions in " << filename << ": "
                  << num_rows << "x" << num_cols << std::endl;
        return false;
    }
    
    // Read image data
    images.reserve(num_images);
    std::vector<unsigned char> pixel_data(kInputSize);
    
    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(pixel_data.data()), kInputSize);
        if (!file) {
            std::cerr << "Error reading image data from " << filename << std::endl;
            return false;
        }
        
        images.push_back(preprocessImage(pixel_data));
    }
    
    return true;
}

bool Dataset::readLabelFile(const std::string& filename, std::vector<Label>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Read the magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = __builtin_bswap32(magic); // Convert from big endian
    
    // Check magic number (2049 for label file)
    if (magic != 2049) {
        std::cerr << "Invalid magic number in label file: " << filename << std::endl;
        return false;
    }
    
    // Read number of labels
    uint32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels); // Convert from big endian
    
    // Read label data
    labels.reserve(num_labels);
    unsigned char label;
    
    for (uint32_t i = 0; i < num_labels; ++i) {
        file.read(reinterpret_cast<char*>(&label), 1);
        if (!file) {
            std::cerr << "Error reading label data from " << filename << std::endl;
            return false;
        }
        
        if (label >= kNumClasses) {
            std::cerr << "Invalid label value: " << static_cast<int>(label) << std::endl;
            return false;
        }
        
        labels.push_back(oneHotEncode(label));
    }
    
    return true;
}

} // namespace mnist
