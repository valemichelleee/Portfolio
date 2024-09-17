// #pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include "../lib/eigen-master/Eigen/Dense"
#include <cmath>
class NeuralNetwork {
public:
    // Constructor 
    NeuralNetwork(int nn_input_size, int nn_hidden_size, int nn_num_classes);

    // Destructor 
    //~NeuralNetwork();

    // Forward pass
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x);

    // Backward pass
    void backward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double learning_rate);

    // Cross Entropy Loss
    double cross_entropy_loss(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels);

private:
    // Network parameters
    int input_size, hidden_size, num_classes;
    
    // Weight matrices and bias vectors
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2;

    // ReLU activation
    Eigen::MatrixXd relu(const Eigen::MatrixXd& x);

    // Softmax function
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);

    // Initialize weight and biases
    void init_weights_and_biases();

    // Backward pass helper functions
    Eigen::MatrixXd softmax_derivative(const Eigen::MatrixXd& x);

    Eigen::MatrixXd refactorLabel(const Eigen::MatrixXd& y, int num_classes);
};

// Constructor 
NeuralNetwork::NeuralNetwork(int nn_input_size, int nn_hidden_size, int nn_num_classes) : input_size(nn_input_size), hidden_size(nn_hidden_size), num_classes(nn_num_classes) {
    // Initialize weights and biases
    init_weights_and_biases();
}

// Initialize weight and biases (-1, 1)
void NeuralNetwork::init_weights_and_biases() {
    // Initialize W1 and W2 with random values
    W1 = Eigen::MatrixXd::Random(input_size, hidden_size);
    W2 = Eigen::MatrixXd::Random(hidden_size, num_classes);

    // Initialize b1 and b2 to zero
    b1 = Eigen::VectorXd::Random(hidden_size);
    b2 = Eigen::VectorXd::Random(num_classes);
}

// Forward pass
Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& x) {
    // First layer
    Eigen::MatrixXd z1 = x * W1;
    z1.rowwise() += b1.transpose(); // Add bias
    Eigen::MatrixXd a1 = relu(z1); // Apply ReLU activation function

    // Second layer
    Eigen::MatrixXd z2 = a1 * W2;
    z2.rowwise() += b2.transpose(); // Add bias
    
    return softmax(z2);
}

Eigen::MatrixXd NeuralNetwork::refactorLabel(const Eigen::MatrixXd& y, int num_classes) {
    int num_samples = y.cols();
    Eigen::MatrixXd one_hot_matrix = Eigen::MatrixXd::Zero(num_samples, num_classes);

    for (int i = 0; i < num_samples; ++i) {
        int label = static_cast<int>(y(i));
        one_hot_matrix(i, label) = 1.0;
    }

    return one_hot_matrix;
}

// Backward pass
void NeuralNetwork::backward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double learning_rate) {
    // Forward pass
    Eigen::MatrixXd z1 = x * W1;
    z1.rowwise() += b1.transpose();
    Eigen::MatrixXd a1 = relu(z1);

    Eigen::MatrixXd z2 = a1 * W2;
    z2.rowwise() += b2.transpose();
    Eigen::MatrixXd output = softmax(z2);

    std::cout << cross_entropy_loss(output, y) << std::endl;

    Eigen::MatrixXd qwer = refactorLabel(y, 10);
    // Compute loss gradient with respect to the output
    Eigen::MatrixXd dz2 = output - refactorLabel(y, 10);

    // Gradient for W2 and b2
    Eigen::MatrixXd dW2 = a1.transpose() * dz2;
    Eigen::VectorXd db2 = dz2.colwise().sum();

    // Gradient for the ReLU activation function
    Eigen::MatrixXd dz1 = dz2 * W2.transpose();
    dz1 = dz1.array() * (z1.array() > 0).cast<double>(); // Element-wise multiplication with ReLU derivative

    // Gradient for W1 and b1
    Eigen::MatrixXd dW1 = x.transpose() * dz1;
    Eigen::VectorXd db1 = dz1.colwise().sum();

    // Update weights and biases using gradient descent
    W1 -= learning_rate * dW1;
    b1 -= learning_rate * db1;
    W2 -= learning_rate * dW2;
    b2 -= learning_rate * db2;
}

// ReLU activation
Eigen::MatrixXd NeuralNetwork::relu(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0.0);
}

// Softmax function
Eigen::MatrixXd NeuralNetwork::softmax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd exp_x = x.unaryExpr([](double val) {return std::exp(val);});
    return exp_x.array().colwise() / exp_x.rowwise().sum().array();
}

// Softmax derivative function to compute the gradient of the loss with respect to the output of the last layer 
Eigen::MatrixXd NeuralNetwork::softmax_derivative(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd softmax_output = softmax(x);
    return softmax_output.array() * (1.0 - softmax_output.array());
}

// Cross Entropy Loss
double NeuralNetwork::cross_entropy_loss (const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels) {
    double loss = 0.0;
    
    for (int i = 0; i < output.rows(); ++i) {
        int label = labels(i);
        double predicted_probability = output(i, label);
        loss += std::log(predicted_probability + 1e-10); // avoiding log(0)
    }

    return loss; // return loss/output.rows() for average loss

}