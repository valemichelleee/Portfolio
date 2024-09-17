# echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."
#!/bin/bash


# Run the exe 
if [ $# -ne 3 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3>"
    exit 1
fi

# Run neural_network.exe with 3 arguments
./neural_network_IO $1 $2 $3

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo "Images reading successful."
else
    echo "Images reading failed."
    exit 1
fi