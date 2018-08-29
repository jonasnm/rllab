#!/bin/bash

# Running experiments with default parameters
#function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/command_line_functions/run_tnpg_no_stub.py
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/

# Saving the plots!
plot_function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/load_and_sim_policy.py
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/TNPG_8_default.png -t 'TNPG 8 hidden units'
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/TNPG_32_32_default.png -t 'TNPG 32, 32 hidden units'
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/TNPG_100_50_25_default.png -t 'TNPG 100, 50, 25 hidden units'

python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/8/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/8/TNPG_8_default.png -t 'TNPG 8 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/32_32/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/32_32/TNPG_32_32_default.png -t 'TNPG 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25/TNPG_100_50_25_default.png -t 'TNPG 100, 50, 25 hidden units'
