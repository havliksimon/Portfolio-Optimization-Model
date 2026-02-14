#!/bin/bash
# Install script for Arch Linux with Python 3.14

echo "Installing Portfolio Optimizer on Arch Linux..."
echo "Python version: $(./venv/bin/python --version)"

# Install system dependencies needed for compilation
sudo pacman -S --needed gcc gcc-fortran python-devel openblas lapack

# Upgrade pip first
./venv/bin/pip install --upgrade pip wheel setuptools

# Install numpy first (required by pandas)
./venv/bin/pip install numpy --no-build-isolation

# Install pandas with no build isolation
./venv/bin/pip install pandas --no-build-isolation

# Install scipy
./venv/bin/pip install scipy --no-build-isolation

# Install remaining requirements
./venv/bin/pip install -r requirements.txt

echo "Installation complete!"
echo "Run: ./venv/bin/flask --app app run --host=0.0.0.0 --port=5000 --debug"
