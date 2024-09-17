// #pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "../lib/eigen-master/Eigen/Dense"
#include "DataIO.hpp"

int main(int argc, char *argv[])
{
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    int required_element_index = atoi(argv[3]);

    DataIO dataIO;
    Eigen::MatrixXd tensor = dataIO.readTensor(input_file);

    if (tensor.size())
    {
        dataIO.writeTensorToFile(tensor, output_file, required_element_index);
    }
    else
    {
        std::cerr << "Error in creating the Tensor" << std::endl;
    }
    return 0;
}
