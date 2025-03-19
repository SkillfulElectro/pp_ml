#include "Layer.h"

/*
This code attempts to experiment with a ping-pong-style neural network.
*/


int main() {
    // Use a lower learning rate to keep updates stable.
    double learning_rate = 0.000001;

    // Define network structure:
    // Layer 1: 1 node, input dimension = 2 (from x1, x2).
    Layer<double> layer1(1, 2, learning_rate);
    // Layer 2: Hidden layer 1 with 3 nodes, input dimension = 1 (from layer 1).
    Layer<double> layer2(3, 1, learning_rate);
    // Layer 3: Hidden layer 2 (extra layer), 3 nodes, input dimension = 3 (from layer 2).
    Layer<double> layer3(3, 3, learning_rate);
    // Layer 4: Output layer with 1 node, input dimension = 3 (from layer 3).
    Layer<double> layer4(1, 3, learning_rate);

    // Training data: target function y = 2*x1 + 3*x2 + 5.
    // Each entry is a pair of inputs {x1, x2}.
    std::vector<std::pair<double, double>> training_inputs = { {-10, -5}, {-5, 0}, {0, 5}, {5, 10}, {10, 15} , {20 , 200} , {30 , 46} };
    int num_epochs = 19000;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (auto& input_pair : training_inputs) {
            double x1 = input_pair.first;
            double x2 = input_pair.second;

            std::vector<double> input = { x1, x2 };

            std::vector<double> out1 = layer1.forward(input);
            std::vector<double> out2 = layer2.forward(out1);
            std::vector<double> out3 = layer3.forward(out2);
            std::vector<double> out4 = layer4.forward(out3);
            double pred = out4[0];


            double target = 2.0 * x1 + 3.0 * x2 + 5.0;
            double error_output = target - pred;


            std::vector<double> error_layer4 = { error_output };
            std::vector<double> grad_layer3 = layer4.backward(error_layer4);


            std::vector<double> error_layer3 = grad_layer3;
            std::vector<double> grad_layer2 = layer3.backward(error_layer3);


            std::vector<double> error_layer2 = grad_layer2;
            std::vector<double> grad_layer1 = layer2.backward(error_layer2);


            std::vector<double> error_layer1 = grad_layer1;
            layer1.backward(error_layer1);
        }

        // print average loss every 500 epochs.
        if (epoch % 500 == 0) {
            double total_loss = 0.0;
            for (auto& input_pair : training_inputs) {
                double x1 = input_pair.first;
                double x2 = input_pair.second;
                std::vector<double> input = { x1, x2 };
                std::vector<double> out1 = layer1.forward(input);
                std::vector<double> out2 = layer2.forward(out1);
                std::vector<double> out3 = layer3.forward(out2);
                std::vector<double> out4 = layer4.forward(out3);
                double pred = out4[0];
                double target = 2.0 * x1 + 3.0 * x2 + 5.0;
                double err = target - pred;
                total_loss += err * err;
            }

            total_loss /= training_inputs.size();
            std::cout << "Epoch " << epoch << " Loss: " << total_loss << std::endl;
        }
    }


    std::cout << "\nLayer 1 parameters:" << std::endl;
    layer1.print_layer();
    std::cout << "\nLayer 2 parameters:" << std::endl;
    layer2.print_layer();
    std::cout << "\nLayer 3 parameters:" << std::endl;
    layer3.print_layer();
    std::cout << "\nLayer 4 parameters:" << std::endl;
    layer4.print_layer();


    std::cout << "\nTesting network:" << std::endl;
    for (double x1{ 0.0 }; x1 < 20.0; ++x1) {
        for (double x2{ 0.0 }; x2 < 30.0; ++x2) {

            std::vector<double> input = { x1, x2 };
            std::vector<double> out1 = layer1.forward(input);
            std::vector<double> out2 = layer2.forward(out1);
            std::vector<double> out3 = layer3.forward(out2);
            std::vector<double> out4 = layer4.forward(out3);

            double pred = out4[0];

            std::cout << "Input: (" << x1 << ", " << x2 << ") -> Prediction: " << pred << " , org : " << x1 * 2 + x2 * 3 + 5 << std::endl;
        }
    }

    return 0;
}
