#!/bin/bash

# Setting different parameters
function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/command_line_functions/run_ddpg_no_stub.py
file_path=/home/jonas/Dropbox/results/jonas_experiments/ddpg/

# Plotting
plot_function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/load_and_sim_policy.py

#=============
# Experiments
#=============

# Gaussian with insulin
#python $function_path HovorkaGaussianInsulin-v0 --hidden_sizes 3 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian_with_insulin/

## Gaussian
#python $function_path HovorkaGaussian-v0        --hidden_sizes 3 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian/

## Absolute
#python $function_path HovorkaAbsolute-v0        --hidden_sizes 3 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/absolute/

## binary
#python $function_path HovorkaBinary-v0          --hidden_sizes 3 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/binary_tight/

#==========
# Plotting
#==========

# Gaussian
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian/params.pkl \
  -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian/ddpg_400_300_gaussian.png -t 'ddpg gaussian 400, 300 hidden units'

# Gaussian with insulin
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian_with_insulin/params.pkl \
  -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/gaussian_with_insulin/ddpg_400_300_gaussian_with_insulin.png -t 'ddpg gaussian with insulin 400, 300 hidden units'

# binary
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/binary_tight/params.pkl \
  #-ff /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/binary_tight/ddpg_400_300_binary.png -t 'ddpg binary_tight 400, 300 hidden units'

# absolute
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg_larger_net/absolute/params.pkl \
  -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/absolute/ddpg_400_300_absolute.png -t 'ddpg absolute 8 hidden units'
