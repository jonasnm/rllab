#!/bin/bash

# Running experiments with default parameters
file_path=/home/jonas/Dropbox/results/jonas_experiments/tnpg/

function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/command_line_functions/run_tnpg_no_stub.py

# Gaussian with insulin
python $function_path HovorkaGaussianInsulin-v0 --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/
#python $function_path HovorkaGaussianInsulin-v0 --hidden_sizes 0 --data_dir $file_path'gaussian_with_insulin/5000/8'
python $function_path HovorkaGaussianInsulin-v0 --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/
python $function_path HovorkaGaussianInsulin-v0 --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/

# Gaussian 
python $function_path HovorkaGaussian-v0        --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/8/
python $function_path HovorkaGaussian-v0        --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/32_32/
python $function_path HovorkaGaussian-v0        --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25/

# Absolute
python $function_path HovorkaAbsolute-v0        --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/8/
python $function_path HovorkaAbsolute-v0        --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/32_32/
python $function_path HovorkaAbsolute-v0        --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/100_50_25/

# binary
python $function_path HovorkaBinary-v0          --hidden_sizes 0 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/8/
python $function_path HovorkaBinary-v0          --hidden_sizes 1 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/32_32/
python $function_path HovorkaBinary-v0          --hidden_sizes 2 --data_dir /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/100_50_25/

# Plotting
plot_function_path=/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/load_and_sim_policy.py


# Gaussian
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/8/params.pkl                      -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/8/tnpg_8_default.png -t 'tnpg gaussian 8 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/32_32/params.pkl                  -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/32_32/tnpg_32_32_default.png -t 'tnpg gaussian 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25/params.pkl              -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25/tnpg_100_50_25_default.png -t 'tnpg gaussian 100, 50, 25 hidden units'

# Gaussian with insulin
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/params.pkl -pp True        -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/8/tnpg_8_default.png -t 'tnpg gaussian with insulin 8 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/params.pkl     -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/32_32/tnpg_32_32_default.png -t 'tnpg gaussian_with_insulin 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/params.pkl -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian_with_insulin/5000/100_50_25/tnpg_100_50_25_default.png -t 'tnpg gaussian with insulin 100, 50, 25 hidden units'

# binary
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/8/params.pkl                  -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/8/tnpg_8_default.png -t 'tnpg binary_tight 8 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/32_32/params.pkl              -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/32_32/tnpg_32_32_default.png -t 'tnpg binary_tight 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/100_50_25/params.pkl          -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/binary_tight/5000/100_50_25/tnpg_100_50_25_default.png -t 'tnpg binary_tight 100, 50, 25 hidden units'

# absolute
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/8/params.pkl                     -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/8/tnpg_8_default.png -t 'tnpg absolute 8 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/32_32/params.pkl                  -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/32_32/tnpg_32_32_default.png -t 'tnpg absolute 32, 32 hidden units'
python $plot_function_path /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/100_50_25/params.pkl              -ff /home/jonas/Dropbox/results/jonas_experiments/tnpg/absolute/5000/100_50_25/tnpg_100_50_25_default.png -t 'tnpg absolute 100, 50, 25 hidden units'
