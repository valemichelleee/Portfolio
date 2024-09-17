# echo "This script should trigger the training and testing of your neural network implementation..."
#!/bin/bash

# Assuming this script is run from the project root directory

# Navigate to the directory containing the MNIST data

#cd mnist-datasets || { echo "MNIST data directory not found"; exit 1; }

# Execute the neural network training and testing
# Assuming neural-net is in the project root directory
#cd ..

# Run the exe 
if [ $# -ne 1 ]; then
    echo "Usage: $0 <arg1>"
    exit 1
fi

# Run neural_network.exe with 1 argument, the relative input path to a configuration file
./neural_network $1

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo "Training and testing successful."
else
    echo "Training and testing failed."
    exit 1
fi