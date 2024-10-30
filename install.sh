#!/bin/bash

# Step 1: Download Isaac Gym Preview 4
echo "Downloading Isaac Gym Preview 4..."
wget https://developer.nvidia.com/isaac-gym-preview-4 -O IsaacGym_Preview_4_Package.tar

# Step 2: Extract the IsaacGym_Preview_4_Package.tar
echo "Extracting Isaac Gym package..."
tar -xf IsaacGym_Preview_4_Package.tar

# Step 3: Delete the tar file after extraction
echo "Deleting the tar file..."
rm IsaacGym_Preview_4_Package.tar

# Step 4: Ask the user for the environment name
echo "Please enter the name for the new conda environment:"
read env_name

# Step 5: Initialize Conda for the current shell session
echo "Initializing Conda for the shell session..."
eval "$(conda shell.bash hook)"

# Step 6: Create the environment using conda
echo "Creating conda environment: $env_name"
conda create -n "$env_name" python=3.8 -y

# Step 7: Activate the environment
echo "Activating conda environment: $env_name"
conda activate "$env_name"

# Step 8: Navigate to the Isaac Gym folder and install using pip
echo "Installing Isaac Gym in editable mode..."
pip install -e isaacgym/python

# Step 9: Navigate to the denoising-diffusion-pytorch folder and install using pip
echo "Installing Denoising Diffusion in editable mode..."
pip install -e contexts/denoising-diffusion-pytorch/

# Step 10: Install Redis server
echo "Installing Redis server..."
sudo apt install redis-server -y

# Step 10: Install Python 3.8 Dev if your system is newer than Ubuntu18?
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install libpython3.8-dev

echo "Installation complete! Isaac Gym and Denoising Diffusion have been set up."