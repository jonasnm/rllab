from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import os

from utils.load_and_sim_policy import render_and_plot_policy
from utils.plot_training_progress import plot_training_progress

# File name for saving
algorithm = 'TRPO'

# models = ('HovorkaAbsolute-v0', 'HovorkaBinary-v0', 'HovorkaGaussian-v0', 'HovorkaGaussianInsulin-v0')
environment = 'HovorkaGaussian-v0'

folder_prefix = './results/'

# ==========================================================================
# OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
# HovorkaIntervalRandom starts at a random value
# ==========================================================================

def run_task(*_):
    ''' Wrapper function for the parameters setup
    and training of the RL algorithm.
    '''
    env = normalize(GymEnv(environment))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    learn_std = True
    init_std = 1
    # hidden_sizes = NN_sizes[i]
    # hidden_sizes=(8,)
    # hidden_sizes=(32, 32)
    hidden_sizes=(100, 50, 25)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        learn_std=learn_std,
        init_std=init_std
    )

    # =======================
    # Defining the algorithm
    # =======================
    batch_size = 5000
    n_itr = 200
    gamma = .99
    step_size = 0.01
    # max_path_length = 96,
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        # max_path_length=max_path_length,
        n_itr=n_itr,
        discount=gamma,
        step_size=step_size
    )
    algo.train()

# NN_folder = [str(j) for j in NN_sizes[i]]
# NN_folder = '_'.join(NN_folder)
nn_folder_name = '100_50_25'
log_dir = folder_prefix + algorithm + '/' + nn_folder_name + '/' + environment

if os.path.isdir('./Documents'):
    log_dir = log_dir + '_1'

# Running and saving the experiment
run_experiment_lite(
    run_task,
    # algo.train(),
    log_dir=log_dir,
    # n_parallel=2,
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # exp_prefix="Reinforce_" + env_name,
    # exp_prefix=data_dir
    plot=False
)

## Testing and plotting the policy
data_dir = log_dir
filename = log_dir + '/params.pkl'
figure_filename = data_dir + '.png'
title = algorithm + '_' + environment + '_' + nn_folder_name

render_and_plot_policy(filename, figure_filename, title)

# Plotting training progres
plot_training_progress(log_dir + '/progress.csv', data_dir + 'training_progress.png', title + ' training progress')
