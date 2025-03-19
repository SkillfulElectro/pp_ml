#ifndef LINEAR_RG
#define LINEAR_RG

#include "headers.h"

// LinearNode: a single linear unit with stored input for backprop.
template <typename T>
class LinearNode {
public:
    T learning_rate;
    std::vector<T> weights;
    T bias;
    std::vector<T> last_input;

    LinearNode() : learning_rate(0), bias(0) { }


    void init_node(size_t num_params, T lr /*learning rate*/) {
        learning_rate = lr;
        weights.resize(num_params);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-0.1, 0.1);

        for (size_t i = 0; i < num_params; i++) {
            weights[i] = dis(gen);
        }
        bias = dis(gen);
    }

    // ping : store the input and compute output = bias + dot(weights, input)
    T ping(const std::vector<T>& x) {
        assert(x.size() == weights.size() && "Input vector size must match weights size.");
        last_input = x;  // save for backpropagation
        T output = bias;
        for (size_t i = 0; i < x.size(); i++) {
            output += weights[i] * x[i];
        }
        return output;
    }

    // pong :
    //   error: the gradient of the loss with respect to this node's output.
    // Updates parameters (weights and bias) using the stored input.
    // Returns the gradient with respect to the input for backpropagation.
    std::vector<T> pong(T error) {

        std::vector<T> old_weights = weights;


        bias += learning_rate * error;
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] += learning_rate * error * last_input[i];
        }


        std::vector<T> grad_input(weights.size(), T(0));
        for (size_t i = 0; i < weights.size(); i++) {
            grad_input[i] = error * old_weights[i];
        }
        return grad_input;
    }
};

#endif // !LINEAR_RG
