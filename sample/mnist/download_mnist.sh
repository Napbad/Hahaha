#!/bin/bash
# Script to download and extract the MNIST dataset

# Create data directory
mkdir -p data
cd data

# Download dataset files
echo "Downloading MNIST dataset..."
wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Extract files
echo "Extracting files..."
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

echo "MNIST dataset downloaded and extracted successfully."
