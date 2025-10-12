#include <iostream>

#include "models/nn/Sequential.h"
#include "models/nn/layers/Linear.h"
#include "optimizers/SGDOptimizer.h"
#include "core/loss/MSELoss.h"
#include "core/compute/Variable.h"

using namespace hahaha::ml;
using namespace hahaha::common::ds;

int main() {
    // 1. Define the Model
    auto model = Sequential<float>();
    model.add(new Linear<float>(1, 1));

    // 2. Create the SGOptimizer
    auto optimizer = SGDOptimizer(model.parameters(), 0.01);

    // 3. Create the Loss Function
    auto loss_fn = MSELoss<float>();

    // 4. Create Synthetic Data (y = 2x + 1)
    Tensor<float> x_train({10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<float> y_train({10, 1}, {1, 3, 5, 7, 9, 11, 13, 15, 17, 19});
    
    auto x_var = Variable(x_train);
    auto y_var = Variable(y_train);

    // 5. Training Loop
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        auto y_pred = model.forward(x_var);

        // Compute loss
        auto loss = loss_fn(y_pred, y_var);

        // Zero gradients
        optimizer.zero_grad();

        // Backward pass (compute gradients)
        loss.backward();

        // Update weights
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.data()[0] << std::endl;
        }
    }
    std::cout << "Training finished." << std::endl;

    // 6. Print final parameters
    auto params = model.parameters();
    // Assuming the first layer is the Linear layer we added
    auto& weight_var = *params[0];
    auto& bias_var = *params[1];

    std::cout << "Learned weights: " << weight_var.data()[0] << std::endl;
    std::cout << "Learned bias: " << bias_var.data()[0] << std::endl;

    return 0;
}
