from rllab.algos.ddpg import DDPG
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, stub
import datetime
import dateutil.tz
from rllab.config import PROJECT_PATH
import sys
sys.path.insert(0, '/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/')
from load_and_sim_policy import render_and_plot_policy

# from plotting import test_and_plot

import os.path as osp

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')


# Running the experiment in "stub" mode -- for use with the
# run_experiment_lite function...
stub(globals())

# ==========================================================================
# OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
# HovorkaIntervalRandom starts at a random value
# ==========================================================================
env_name ='HovorkaInterval-v0'
# env_name = 'HovorkaIntervalRandom-v0'

env = normalize(GymEnv(env_name))


# ==============================
# Changing the reward function
# ==============================
# reward_fun = 'gaussian_with_insulin'
reward_fun = 'gaussian'
env.wrapped_env.env.env.reward_flag = reward_fun

# ======================
# Experiment parameters
# ======================
baseline = LinearFeatureBaseline(env_spec=env.spec)

batch_size = 32
n_itr = 1000
gamma = .9
step_size = 0.01
learn_std = True
init_std=1

# =========================================
# Setting the neural network architecture
# =========================================
# hidden_sizes=(8,)
hidden_sizes=(32, 32)
# hidden_sizes=(100, 50, 25)

# ===================
# Defining the policy
# ===================
policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=hidden_sizes,
)

# =======================
# Defining the algorithm
# =======================
es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec)

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    max_path_length=96,
    epoch_length=1000,
    min_pool_size=10000,
    batch_size=batch_size,
    discount=gamma,
    n_epochs=n_itr,
    scale_reward=0.01,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
)


# Formatting string for data directory
hidden_arc = [str(i) for i in hidden_sizes]
hidden_arc = '_'.join(hidden_arc)

data_dir = 'DDPG_{}_nIters_{}_stepSize_{}_gamma_{}_initStd_{}{}_policyPar_{}_reward_{}'\
        .format(batch_size, n_itr, step_size,''.join(str(gamma).split('.')), init_std, learn_std, hidden_arc, reward_fun)

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

DROPBOX_DIR = '/home/jonas/Dropbox/results/jonas_experiments/'
# log_dir = PROJECT_PATH + '/data/local/' + data_dir + timestamp
log_dir = DROPBOX_DIR + data_dir + timestamp

# Running and saving the experiment
run_experiment_lite(
    algo.train(),
    log_dir='.',
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # exp_prefix="Reinforce_" + env_name,
    # exp_prefix=data_dir
    seed=1,
    mode="local",
    plot=False,
    # terminate_machine=args.dont_terminate_machine,
    # added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)


## Testing the policy
filename = log_dir + '/params.pkl'
figure_filename = DROPBOX_DIR + data_dir  + '.png'

title_params = {
    'batch_size': batch_size,
    'n_itr': n_itr,
    'step_size': step_size,
    'gamma': gamma,
    'init_std': init_std,
    'learn_std': learn_std,
    'hidden_arc': hidden_arc,
    'reward_fun': reward_fun,
}

render_and_plot_policy('DDPG', filename, figure_filename, title_params)

