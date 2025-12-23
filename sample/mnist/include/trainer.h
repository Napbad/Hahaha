#pragma once

#include <string>
#include <functional>
#include "network.h"
#include "dataset.h"

namespace mnist {

/**
 * @brief Configuration for training process
 */
struct TrainingConfig {
    size_t epochs = 10;             ///< Number of training epochs
    size_t batch_size = 64;         ///< Batch size for training
    float learning_rate = 0.001;    ///< Initial learning rate
    bool learning_rate_decay = true; ///< Whether to use learning rate decay
    float lr_decay_rate = 0.5;      ///< Learning rate decay factor
    size_t lr_decay_steps = 3;      ///< Epochs between learning rate decay
    std::string checkpoint_dir = ""; ///< Directory to save model checkpoints
    bool verbose = true;            ///< Whether to print training progress
};

/**
 * @brief Callback interface for monitoring training process
 */
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    
    /**
     * @brief Called at the beginning of training
     * 
     * @param config Training configuration
     */
    virtual void onTrainingBegin(const TrainingConfig& config) {}
    
    /**
     * @brief Called at the beginning of each epoch
     * 
     * @param epoch Current epoch number
     */
    virtual void onEpochBegin(size_t epoch) {}
    
    /**
     * @brief Called at the end of each epoch
     * 
     * @param epoch Current epoch number
     * @param metrics Dictionary of metrics for this epoch (e.g., loss, accuracy)
     */
    virtual void onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) {}
    
    /**
     * @brief Called at the end of training
     * 
     * @param metrics Final metrics
     */
    virtual void onTrainingEnd(const std::pair<float, float>& metrics) {}
};

/**
 * @brief Simple console logger callback
 */
class ConsoleLoggerCallback : public TrainingCallback {
public:
    void onTrainingBegin(const TrainingConfig& config) override;
    void onEpochBegin(size_t epoch) override;
    void onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) override;
    void onTrainingEnd(const std::pair<float, float>& metrics) override;
};

/**
 * @brief Model checkpoint callback
 */
class ModelCheckpointCallback : public TrainingCallback {
public:
    /**
     * @brief Construct a ModelCheckpointCallback
     * 
     * @param directory Directory to save checkpoints
     * @param save_best_only Only save if model has improved
     * @param monitor Metric to monitor ('loss' or 'accuracy')
     */
    ModelCheckpointCallback(const std::string& directory, 
                           bool save_best_only = true, 
                           const std::string& monitor = "accuracy");
    
    void onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) override;
    
private:
    std::string m_directory;
    bool m_save_best_only;
    std::string m_monitor;
    float m_best_value = 0.0f;
    Network* m_network = nullptr;
};

/**
 * @brief Early stopping callback
 */
class EarlyStoppingCallback : public TrainingCallback {
public:
    /**
     * @brief Construct an EarlyStoppingCallback
     * 
     * @param patience Number of epochs with no improvement before stopping
     * @param min_delta Minimum change to qualify as improvement
     * @param monitor Metric to monitor ('loss' or 'accuracy')
     */
    EarlyStoppingCallback(size_t patience = 5, 
                         float min_delta = 0.001, 
                         const std::string& monitor = "loss");
    
    void onEpochEnd(size_t epoch, const std::pair<float, float>& metrics) override;
    
    /**
     * @brief Check if training should stop
     * 
     * @return true if training should stop
     */
    bool shouldStop() const { return m_should_stop; }
    
private:
    size_t m_patience;
    float m_min_delta;
    std::string m_monitor;
    size_t m_wait = 0;
    float m_best_value = 0.0f;
    bool m_should_stop = false;
};

/**
 * @brief Trainer class for training neural networks on MNIST dataset
 */
class Trainer {
public:
    /**
     * @brief Construct a Trainer object
     * 
     * @param network Neural network to train
     * @param dataset MNIST dataset
     */
    Trainer(Network& network, const Dataset& dataset);
    
    /**
     * @brief Train the network
     * 
     * @param config Training configuration
     * @return std::pair<float, float> Final (loss, accuracy) on test set
     */
    std::pair<float, float> train(const TrainingConfig& config);
    
    /**
     * @brief Add a callback for monitoring training process
     * 
     * @param callback Callback object
     */
    void addCallback(std::shared_ptr<TrainingCallback> callback);
    
private:
    Network& m_network;
    const Dataset& m_dataset;
    std::vector<std::shared_ptr<TrainingCallback>> m_callbacks;
    
    /**
     * @brief Train one epoch
     * 
     * @param batch_size Batch size for training
     * @return Average loss for this epoch
     */
    float trainEpoch(size_t batch_size);
};

} // namespace mnist
