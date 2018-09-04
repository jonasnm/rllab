#!/bin/bash
echo "Creating conda environment..."
conda env create -f environment.yml
conda env update

source activate rllab3
conda install nomkl numpy scipy scikit-learn numexpr
conda remove mkl mkl-service
