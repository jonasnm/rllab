#!/bin/bash
source activate rllab3
conda install nomkl numpy scipy scikit-learn numexpr
conda remove mkl mkl-service
