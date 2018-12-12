from rllab.algos.tnpg import TNPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
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
reward_fun = 'binary_tight'
# reward_fun = 'absolute'
env.wrapped_env.env.env.reward_flag = reward_fun

# ======================
# Experiment parameters
# ======================
baseline = LinearFeatureBaseline(env_spec=env.spec)

batch_size = 5000
n_itr = 200
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
policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=hidden_sizes,
    learn_std=learn_std,
    init_std=init_std
)

# =======================
# Defining the algorithm
# =======================
algo = TNPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=batch_size,
    n_itr=n_itr,
    discount=gamma,
    plot=True,
    step_size=step_size
)

# Formatting string for data directory
hidden_arc = [str(i) for i in hidden_sizes]
hidden_arc = '_'.join(hidden_arc)

data_dir = 'tnpg_{}_nIters_{}_stepSize_{}_gamma_{}_initStd_{}{}_policyPar_{}_reward_{}'\
        .format(batch_size, n_itr, step_size,''.join(str(gamma).split('.')), init_std, learn_std, hidden_arc, reward_fun)

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

DROPBOX_DIR = '/home/jonas/Dropbox/results/jonas_experiments/'
# log_dir = PROJECT_PATH + '/data/local/' + data_dir + timestamp
log_dir = DROPBOX_DIR + data_dir + timestamp

# Running and saving the experiment
run_experiment_lite(
    algo.train(),
    log_dir=log_dir,
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
    # plot=True,
    # terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
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

render_and_plot_policy('TNPG', filename, figure_filename, title_params)

