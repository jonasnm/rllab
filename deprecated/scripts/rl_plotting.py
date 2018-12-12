# from rllab.algos.vpg import VPG
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import sys
sys.path.insert(0, '/Users/jonas/Documents/git/rllab/diabetes_experiments/')
# sys.path.insert(0, '/home/jonas/Documents/git/rllab/diabetes_experiments/')
from load_and_sim_policy import render_and_plot_policy
from plot_training_progress import plot_training_progress

# File name for saving
RL = 'TRPO'

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

models = ('HovorkaAbsolute-v0', 'HovorkaBinary-v0', 'HovorkaGaussian-v0', 'HovorkaGaussianInsulin-v0')
# models = ('CambridgeAbsolute-v0','CambridgeBinary-v0', 'CambridgeGaussian-v0', 'CambridgeGaussianInsulin-v0')

NN_folder = '100_50_25'

        log_dir = '/Users/jonas/Dropbox/results/miguel_experiments/hovorkaCGM/' + RL + '/' + models[k]
        ## Testing the policy
        data_dir = '/Users/jonas/Dropbox/results/miguel_experiments/hovorkaCGM/' + RL + '/' + models[0] + '/' + RL + '_default'

        filename = log_dir + '/params.pkl'
        figure_filename = data_dir + '.png'
        title = RL + '_' + models[k] + '_' + NN_folder

        render_and_plot_policy(filename, figure_filename, title)

        # TODO: Add training progress!
        plot_training_progress(log_dir + '/progress.csv', data_dir + 'training_progress.png', title + 'training progress')
