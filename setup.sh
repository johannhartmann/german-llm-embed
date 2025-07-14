#!/bin/bash

echo "=== German LLM Embedding Model Setup ==="
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n germanembed python=3.12 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate germanembed

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# LLM2Vec is now part of this repository with our modifications
echo "Using integrated LLM2Vec with SmolLM3 support..."

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/{datasets,models,cache}
mkdir -p src/utils

echo
echo "=== Setup Complete ==="
echo
echo "To get started:"
echo "1. conda activate germanembed"
echo "2. python src/scripts/setup_model.py \"HuggingFaceTB/SmolLM3-3B\""
echo "3. python src/scripts/build_dataset.py --language de"
echo "4. python src/scripts/train.py \"data/models/huggingfacetb-smollm3-3b-bi-init\" --stage all"
echo ""
echo "See README.md for more examples and options"