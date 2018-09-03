# Plotting training progress of all experiments

find /home/jonas/Dropbox/results/jonas_experiments/tnpg/gaussian/5000/100_50_25 -name 'progress.csv' \
  -exec python /home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/plot_training_progress.py {} ;
