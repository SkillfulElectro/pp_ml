#pragma once
#ifndef LAYER_ML
#define LAYER_ML



#include "linear.h"

// Layer: a fully connected layer composed of multiple LinearNodes.
template <typename T>
class Layer {
public:
    std::vector<LinearNode<T>> nodes;

    // Create a layer with 'num_nodes' nodes; each node expects an input of dimension 'input_dim'
    Layer(size_t num_nodes, size_t input_dim, T lr) {
        nodes.resize(num_nodes);
        for (auto& node : nodes) {
            node.init_node(input_dim, lr);
        }
    }

    // ping caller
    std::vector<T> forward(const std::vector<T>& input) {
        std::vector<T> outputs;
        for (auto& node : nodes) {
            outputs.push_back(node.ping(input));
        }
        return outputs;
    }

    // pong caller :
    //   error_vector: a vector of error signals (one per node) for this layer.
    // Each node updates its parameters and returns a grad_input vector (w.r.t. its input).
    // We aggregate (sum) those grad_input vectors element‑wise to form the overall gradient 
    // with respect to this layer's input.
    std::vector<T> backward(const std::vector<T>& error_vector) {
        assert(error_vector.size() == nodes.size() && "Error vector size must match number of nodes.");
        size_t input_dim = nodes[0].last_input.size();
        std::vector<T> aggregated_grad(input_dim, T(0));

        for (size_t i = 0; i < nodes.size(); i++) {
            std::vector<T> grad_input = nodes[i].pong(error_vector[i]);
            // Sum up the gradients from all nodes elementwise.
            for (size_t j = 0; j < input_dim; j++) {
                aggregated_grad[j] += grad_input[j];
            }
        }
        return aggregated_grad;
    }

    // Utility: print each node’s bias and weights.
    void print_layer() const {
        for (size_t i = 0; i < nodes.size(); i++) {
            std::cout << " Node " << i << " | Bias: " << nodes[i].bias << " | Weights: ";
            for (auto w : nodes[i].weights) {
                std::cout << w << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif // !LAYER_ML
