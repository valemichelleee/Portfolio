#include "neural_network.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include "DataIO.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iomanip>

int main(int argc, char* argv[]) {
    //------READ_CONFIG_FILE--------------------------------------------------------------------------------------------------------
    // Verify if an argument was provided for config file path
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <relative_path_to_config>" << std::endl;
        return 1;
    }

    std::ifstream configFile(argv[1]);

    if (!configFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    int num_epochs = 0;
    int batch_size = 0;
    int hidden_size = 0;
    double learning_rate = 0.0;
    std::string rel_path_train_images, rel_path_train_labels, rel_path_test_images, rel_path_test_labels, rel_path_log_file;

    std::string key, value, line;
    char equalSign;

    while (std::getline(configFile, line)) {
        std::istringstream iss(line);
        if (iss >> key >> equalSign >> value) {
            try {
                if (key == "num_epochs") {
                    num_epochs = std::stoi(value);
                } else if (key == "batch_size") {
                    batch_size = std::stoi(value);
                } else if (key == "hidden_size") {
                    hidden_size = std::stoi(value);
                } else if (key == "learning_rate") {
                    learning_rate = std::stod(value);
                } else if (key == "rel_path_train_images") {
                    rel_path_train_images = value;
                } else if (key == "rel_path_train_labels") {
                    rel_path_train_labels = value;
                } else if (key == "rel_path_test_images") {
                    rel_path_test_images = value;
                } else if (key == "rel_path_test_labels") {
                    rel_path_test_labels = value;
                } else if (key == "rel_path_log_file") {
                    rel_path_log_file = value;
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error converting value for key " << key << ": " << e.what() << std::endl;
            }
        }
    }

    configFile.close();

    //------CONFIGURE_AND_TRAIN_NEURAL_NETWORK--------------------------------------------------------------------------------------------------------
    
    //Read images and labels
    DataIO dataIO;
    Eigen::MatrixXd train_images = dataIO.readTensor(rel_path_train_images);
    Eigen::MatrixXd train_labels = dataIO.readTensor(rel_path_train_labels);

    Eigen::MatrixXd test_images = dataIO.readTensor(rel_path_test_images);
    Eigen::MatrixXd test_labels = dataIO.readTensor(rel_path_test_labels);

    // Construct object NeuralNetwork
    NeuralNetwork neuralNetwork(train_images.cols(), hidden_size, 10);
    
    // Train neural network
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Calculul numărului total de batch-uri
        int numberOfBatches = train_images.rows() / batch_size;
        std::cout << "Epoch " << epoch + 1 << std::endl;

        // Împărțirea matricei în batch-uri
        for (int batch = 0; batch < numberOfBatches; ++batch) {
            // Afișează batch-ul curent
            std::cout << "Batch" << batch + 1 << ": ";

            // Calculul indecșilor de start și de sfârșit pentru batch-ul curent
            int start_index = batch * batch_size;
            int end_index = (batch + 1) * batch_size;

            // Extrage submatricea corespunzătoare batch-ului curent
            Eigen::MatrixXd batch_matrice = train_images.block(start_index, 0, batch_size, train_images.cols());
            Eigen::MatrixXd batch_etichete = train_labels.block(0, start_index, train_labels.rows(), batch_size);
            neuralNetwork.backward(batch_matrice, batch_etichete, learning_rate);
        }
    }

    //------TEST_NEURAL_NETWORK--------------------------------------------------------------------------------------------------------
    // Test network
    std::ofstream logFile(rel_path_log_file);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << rel_path_log_file << std::endl;
        return 1;
    }

    // Find results with the new weights and biases
    Eigen::MatrixXd predictions = neuralNetwork.forward(test_images);
    Eigen::VectorXi maxIndices(predictions.rows());

    for (int i = 0; i < predictions.rows(); ++i) {
        double max_value = predictions(i, 0);
        int max_index = 0;

        for (int j = 0; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_value) {
                max_value = predictions(i, j);
                max_index = j;
            }
        }

        maxIndices(i) = max_index;
        
    }
    
    int goodPredictions = 0;
    int totalPredictions = test_images.rows();

    for (int i = 0; i < test_images.rows(); ++i) {
        if(i % batch_size == 0)
            logFile << "Current batch: " << i / batch_size << std::endl;
        if(maxIndices(i) == test_labels(i))
            goodPredictions++;
        logFile << " - image " << i << ": Prediction=" << maxIndices(i) << ". Label=" << test_labels(i) << std::endl;
    }
    
    // Show the result in a pretty way
    double successRate = (static_cast<double>(goodPredictions) / totalPredictions) * 100;
    std::cout << "Acuracy: " << std::fixed << std::setprecision(2) << successRate << "%" << std::endl;

    logFile.close();

    return 0;
}
