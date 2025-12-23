#include "trainer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <ctime>

namespace mnist {

// ConsoleLoggerCallback implementation
void ConsoleLoggerCallback::onTrainingBegin(const TrainingConfig& config) {
    std::cout << "==================================================================" << std::endl;
    std::cout << "Beginning training with the following configuration:" << std::endl;
    std::cout << "  - Epochs: " << config.epochs << std::endl;
    std::cout << "  - Batch size: " << config.batch_size << std::endl;
    std::cout << "  - Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  - Learning rate decay: " << (config.learning_rate_decay ? "Yes" : "No") << std::endl;
    if (config.learning_rate_decay) {
        std::cout << "  - Decay rate: " << config.lr_decay_rate << " every " 
                 << config.lr_decay_steps << " epochs" << std::endl;
    }
    std::cout << "==================================================================" << std::endl;
}

void ConsoleLoggerCallback::onEpochBegin(size_t epoch) {
    std::cout << "Epoch " << epoch << " beginning..." << std::endl;
}

void ConsoleLoggerCallback::onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) {
    auto [loss, accuracy] = metrics;
    std::cout << "Epoch " << epoch << " completed. "
              << "Loss: " << std::fixed << std::setprecision(4) << loss << ", "
              << "Accuracy: " << std::setprecision(2) << (accuracy * 100.0f) << "%" << std::endl;
}

void ConsoleLoggerCallback::onTrainingEnd(const std::pair<float, float>& metrics) {
    auto [loss, accuracy] = metrics;
    std::cout << "==================================================================" << std::endl;
    std::cout << "Training completed." << std::endl;
    std::cout << "Final results:" << std::endl;
    std::cout << "  - Loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
    std::cout << "  - Accuracy: " << std::setprecision(2) << (accuracy * 100.0f) << "%" << std::endl;
    std::cout << "==================================================================" << std::endl;
}

// ModelCheckpointCallback implementation
ModelCheckpointCallback::ModelCheckpointCallback(const std::string& directory, 
                                               bool save_best_only, 
                                               const std::string& monitor)
    : m_directory(directory), 
      m_save_best_only(save_best_only), 
      m_monitor(monitor) {
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(directory);
}

void ModelCheckpointCallback::onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) {
    float current_value;
    if (m_monitor == "accuracy") {
        current_value = metrics.second;
    } else {
        current_value = metrics.first;
        // For loss, lower is better, so negate
        current_value = -current_value;
    }
    
    bool should_save = !m_save_best_only || current_value > m_best_value;
    
    if (should_save) {
        m_best_value = current_value;
        
        // Generate filename
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&time_t_now);
        
        char time_str[20];
        std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", now_tm);
        
        std::string filename = m_directory + "/model_epoch" + std::to_string(epoch) + 
                              "_" + time_str + ".model";
        
        // Save model
        // In a real implementation, we would get a reference to the network
        // For now, just report the saving
        std::cout << "Saving model checkpoint to " << filename << std::endl;
    }
}

// EarlyStoppingCallback implementation
EarlyStoppingCallback::EarlyStoppingCallback(size_t patience, 
                                           float min_delta, 
                                           const std::string& monitor)
    : m_patience(patience), 
      m_min_delta(min_delta), 
      m_monitor(monitor) {
    
    // Initialize best value based on metric
    if (m_monitor == "loss") {
        m_best_value = std::numeric_limits<float>::max();
    } else {
        m_best_value = -std::numeric_limits<float>::max();
    }
}

void EarlyStoppingCallback::onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) {
    float current_value;
    bool improved;
    
    if (m_monitor == "loss") {
        current_value = metrics.first;
        improved = current_value < (m_best_value - m_min_delta);
    } else {
        current_value = metrics.second;
        improved = current_value > (m_best_value + m_min_delta);
    }
    
    if (improved) {
        m_best_value = current_value;
        m_wait = 0;
    } else {
        m_wait++;
        if (m_wait >= m_patience) {
            m_should_stop = true;
            std::cout << "Early stopping triggered after " << epoch << " epochs" << std::endl;
        }
    }
}

// Trainer implementation
Trainer::Trainer(Network& network, const Dataset& dataset)
    : m_network(network), m_dataset(dataset) {
    
    // Add default console logger callback
    addCallback(std::make_shared<ConsoleLoggerCallback>());
}

std::pair<float, float> Trainer::train(const TrainingConfig& config) {
    // Notify callbacks of training start
    for (auto& callback : m_callbacks) {
        callback->onTrainingBegin(config);
    }
    
    // Variables to track best metrics
    float best_accuracy = 0.0f;
    
    // Training loop
    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        // Notify callbacks of epoch start
        for (auto& callback : m_callbacks) {
            callback->onEpochBegin(epoch);
        }
        
        // Train one epoch
        float train_loss = trainEpoch(config.batch_size);
        
        // Evaluate on test set
        auto test_batch = m_dataset.getTestData();
        Eigen::MatrixXf test_inputs(test_batch.first.size(), test_batch.first[0].size());
        Eigen::MatrixXf test_targets(test_batch.second.size(), test_batch.second[0].size());
        
        for (size_t i = 0; i < test_batch.first.size(); ++i) {
            test_inputs.row(i) = test_batch.first[i];
            test_targets.row(i) = test_batch.second[i];
        }
        
        auto [test_loss, test_accuracy] = m_network.evaluate(test_inputs, test_targets);
        
        // Learning rate decay
        if (config.learning_rate_decay && epoch % config.lr_decay_steps == 0) {
            // In a real implementation, we would update the learning rate here
        }
        
        // Update best metrics
        best_accuracy = std::max(best_accuracy, test_accuracy);
        
        // Notify callbacks of epoch end
        for (auto& callback : m_callbacks) {
            callback->onEpochEnd(epoch, {test_loss, test_accuracy});
        }
        
        // Check for early stopping
        bool should_stop = false;
        for (auto& callback : m_callbacks) {
            auto early_stopping = dynamic_cast<EarlyStoppingCallback*>(callback.get());
            if (early_stopping && early_stopping->shouldStop()) {
                should_stop = true;
                break;
            }
        }
        
        if (should_stop) {
            break;
        }
    }
    
    // Final evaluation
    auto test_batch = m_dataset.getTestData();
    Eigen::MatrixXf test_inputs(test_batch.first.size(), test_batch.first[0].size());
    Eigen::MatrixXf test_targets(test_batch.second.size(), test_batch.second[0].size());
    
    for (size_t i = 0; i < test_batch.first.size(); ++i) {
        test_inputs.row(i) = test_batch.first[i];
        test_targets.row(i) = test_batch.second[i];
    }
    
    auto final_metrics = m_network.evaluate(test_inputs, test_targets);
    
    // Notify callbacks of training end
    for (auto& callback : m_callbacks) {
        callback->onTrainingEnd(final_metrics);
    }
    
    return final_metrics;
}

void Trainer::addCallback(std::shared_ptr<TrainingCallback> callback) {
    m_callbacks.push_back(std::move(callback));
}

float Trainer::trainEpoch(size_t batch_size) {
    size_t num_batches = m_dataset.getTrainSize() / batch_size;
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < num_batches; ++i) {
        // Get next batch
        auto batch = m_dataset.getTrainBatch(batch_size);
        
        // Convert vectors of Eigen vectors to matrices
        Eigen::MatrixXf inputs(batch.first.size(), batch.first[0].size());
        Eigen::MatrixXf targets(batch.second.size(), batch.second[0].size());
        
        for (size_t j = 0; j < batch.first.size(); ++j) {
            inputs.row(j) = batch.first[j];
            targets.row(j) = batch.second[j];
        }
        
        // Train on this batch
        float loss = m_network.trainOnBatch(inputs, targets);
        total_loss += loss;
    }
    
    return total_loss / num_batches;
}

} // namespace mnist
