#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "network.h"
#include "dataset.h"
#include "trainer.h"

/**
 * @brief Print command line usage information
 * 
 * @param program_name Name of the program executable
 */
void printUsage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data-dir=PATH       Path to the MNIST data directory" << std::endl;
    std::cout << "  --epochs=N            Number of training epochs (default: 10)" << std::endl;
    std::cout << "  --batch-size=N        Batch size for training (default: 64)" << std::endl;
    std::cout << "  --learning-rate=N     Initial learning rate (default: 0.001)" << std::endl;
    std::cout << "  --checkpoint-dir=PATH Directory to save model checkpoints (default: none)" << std::endl;
    std::cout << "  --load-model=PATH     Path to load a pre-trained model (default: none)" << std::endl;
    std::cout << "  --save-model=PATH     Path to save the final model (default: none)" << std::endl;
    std::cout << "  --help                Display this help message and exit" << std::endl;
}

/**
 * @brief Parse command line arguments
 * 
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @param config Output training configuration
 * @param data_dir Output data directory path
 * @param load_model Output path to load model from
 * @param save_model Output path to save model to
 * @return true if arguments were parsed successfully, false if should exit
 */
bool parseArgs(int argc, char* argv[], mnist::TrainingConfig& config, std::string& data_dir,
               std::string& load_model, std::string& save_model) {
    
    // Default values
    data_dir = "./data";
    load_model = "";
    save_model = "";
    config.epochs = 10;
    config.batch_size = 64;
    config.learning_rate = 0.001f;
    config.checkpoint_dir = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else if (arg.find("--data-dir=") == 0) {
            data_dir = arg.substr(11);
        } else if (arg.find("--epochs=") == 0) {
            config.epochs = std::stoi(arg.substr(9));
        } else if (arg.find("--batch-size=") == 0) {
            config.batch_size = std::stoi(arg.substr(13));
        } else if (arg.find("--learning-rate=") == 0) {
            config.learning_rate = std::stof(arg.substr(15));
        } else if (arg.find("--checkpoint-dir=") == 0) {
            config.checkpoint_dir = arg.substr(17);
        } else if (arg.find("--load-model=") == 0) {
            load_model = arg.substr(13);
        } else if (arg.find("--save-model=") == 0) {
            save_model = arg.substr(13);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Print pretty progress bar
 * 
 * @param progress Progress value (0-1)
 * @param width Width of the progress bar
 */
void printProgressBar(float progress, int width = 50) {
    int pos = static_cast<int>(width * progress);
    
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "%\r";
    std::cout.flush();
}

/**
 * @brief Display an MNIST image in the console
 * 
 * @param image Vector containing the image data
 * @param digit Predicted or actual digit
 */
void displayImage(const Eigen::VectorXf& image, int digit) {
    const int size = static_cast<int>(std::sqrt(image.size()));
    
    std::cout << "Digit: " << digit << std::endl;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float pixel = image(i * size + j);
            char c;
            if (pixel < 0.1f) c = ' ';
            else if (pixel < 0.3f) c = '.';
            else if (pixel < 0.5f) c = '-';
            else if (pixel < 0.7f) c = '+';
            else if (pixel < 0.9f) c = '#';
            else c = '@';
            
            std::cout << c << ' ';
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Main function
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return int Exit code
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    mnist::TrainingConfig config;
    std::string data_dir, load_model, save_model;
    
    if (!parseArgs(argc, argv, config, data_dir, load_model, save_model)) {
        return 1;
    }
    
    try {
        // Create directory for checkpoints if specified
        if (!config.checkpoint_dir.empty()) {
            std::filesystem::create_directories(config.checkpoint_dir);
        }
        
        // Load MNIST dataset
        std::cout << "Loading MNIST dataset from " << data_dir << "..." << std::endl;
        mnist::Dataset dataset(data_dir);
        
        // Create and configure neural network
        mnist::Network network;
        
        // Check if we're loading a pre-trained model
        if (!load_model.empty()) {
            std::cout << "Loading model from " << load_model << "..." << std::endl;
            if (!network.load(load_model)) {
                std::cerr << "Failed to load model." << std::endl;
                return 1;
            }
        } else {
            // Define network architecture
            std::cout << "Creating neural network..." << std::endl;
            
            // For MNIST: 784 input features (28x28 pixels), 10 output classes (digits 0-9)
            network.addLayer(mnist::Dataset::kInputSize, 128, mnist::ActivationType::ReLU)
                   .addLayer(128, 64, mnist::ActivationType::ReLU)
                   .addLayer(64, mnist::Dataset::kNumClasses, mnist::ActivationType::Softmax)
                   .compile(mnist::LossType::CategoricalCrossEntropy, 
                            mnist::OptimizerType::Adam, 
                            config.learning_rate);
            
            // Print network summary
            network.summary();
            
            // Create trainer with callbacks
            mnist::Trainer trainer(network, dataset);
            
            // Add model checkpoint callback if requested
            if (!config.checkpoint_dir.empty()) {
                trainer.addCallback(std::make_shared<mnist::ModelCheckpointCallback>(
                    config.checkpoint_dir, true, "accuracy"));
            }
            
            // Add early stopping callback
            trainer.addCallback(std::make_shared<mnist::EarlyStoppingCallback>(5, 0.001f, "loss"));
            
            // Train the network
            std::cout << "Training the network..." << std::endl;
            auto [final_loss, final_accuracy] = trainer.train(config);
            
            // Save the model if requested
            if (!save_model.empty()) {
                std::cout << "Saving model to " << save_model << "..." << std::endl;
                if (!network.save(save_model)) {
                    std::cerr << "Failed to save model." << std::endl;
                    return 1;
                }
            }
        }
        
        // Display some test results
        std::cout << "Testing the network on some examples..." << std::endl;
        auto test_batch = dataset.getTestData();
        const size_t num_samples = 5; // Number of samples to display
        
        // Get random indices
        std::vector<size_t> indices;
        for (size_t i = 0; i < num_samples && i < test_batch.first.size(); ++i) {
            indices.push_back(i);
        }
        
        // Create input matrix for prediction
        Eigen::MatrixXf test_inputs(indices.size(), test_batch.first[0].size());
        for (size_t i = 0; i < indices.size(); ++i) {
            test_inputs.row(i) = test_batch.first[indices[i]];
        }
        
        // Predict digits
        Eigen::VectorXi predictions = network.predict(test_inputs);
        
        // Display results
        for (size_t i = 0; i < indices.size(); ++i) {
            Eigen::Index actual_digit;
            test_batch.second[indices[i]].maxCoeff(&actual_digit);
            
            std::cout << "Sample " << (i + 1) << ":" << std::endl;
            std::cout << "Predicted: " << predictions(i) 
                      << ", Actual: " << actual_digit << std::endl;
            displayImage(test_batch.first[indices[i]], predictions(i));
            std::cout << std::endl;
        }
        
        // Final evaluation
        auto full_test_inputs = Eigen::MatrixXf(test_batch.first.size(), test_batch.first[0].size());
        auto full_test_targets = Eigen::MatrixXf(test_batch.second.size(), test_batch.second[0].size());
        
        for (size_t i = 0; i < test_batch.first.size(); ++i) {
            full_test_inputs.row(i) = test_batch.first[i];
            full_test_targets.row(i) = test_batch.second[i];
        }
        
        auto [test_loss, test_accuracy] = network.evaluate(full_test_inputs, full_test_targets);
        
        std::cout << "Final test results:" << std::endl;
        std::cout << "  - Loss: " << std::fixed << std::setprecision(4) << test_loss << std::endl;
        std::cout << "  - Accuracy: " << std::setprecision(2) << (test_accuracy * 100.0f) << "%" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
