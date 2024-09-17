#!/bin/bash

# Change directory to 'src'
cd src

# Compile the source files
g++ -std=c++20 test_DataIO.cpp -o neural_network_IO
g++ -std=c++20 -O3 -march=native test_NeuralNetword.cpp -o neural_network


# Move the executables to the project's main directory
mv neural_network_IO ../
mv neural_network ../

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation failed."
    exit 1
fi