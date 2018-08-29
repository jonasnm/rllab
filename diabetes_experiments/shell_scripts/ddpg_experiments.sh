#!/bin/bash

# Setting different parameters
function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/command_line_functions/run_ddpg_no_stub.py
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/8/
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/32_32/
#python $function_path HovorkaInterval-v0 --reward gaussian_with_insulin --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/100_50_25/

#python $function_path HovorkaInterval-v0 --reward gaussian --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/8/
#python $function_path HovorkaInterval-v0 --reward gaussian --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/32_32/
#python $function_path HovorkaInterval-v0 --reward gaussian --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/100_50_25/

# Plotting
plot_function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/load_and_sim_policy.py
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/8/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/8/ddpg_8_default.png -t 'ddpg gaussian 8 hidden units'
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/32_32/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/32_32/ddpg_32_32_default.png -t 'ddpg gaussian 32, 32 hidden units'
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/100_50_25/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian/5000/100_50_25/ddpg_100_50_25_default.png -t 'ddpg gaussian 100, 50, 25 hidden units'

#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/8/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/8/ddpg_8_default.png -t 'ddpg gaussian_with_insulin 8 hidden units'
#python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/32_32/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/32_32/ddpg_32_32_default.png -t 'ddpg gaussian_with_insulin 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/100_50_25/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/ddpg/gaussian_with_insulin/5000/100_50_25/ddpg_100_50_25_default.png -t 'ddpg gaussian 100, 50, 25 hidden units'
