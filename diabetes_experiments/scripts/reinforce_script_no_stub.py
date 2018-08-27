from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import sys
# sys.path.insert(0, '/home/jonas/Documents/git/EXTERNAL/rllab_fork/diabetes_experiments/')
sys.path.insert(0, '/Users/jonas/Documents/git/rllab/diabetes_experiments/')
from load_and_sim_policy import render_and_plot_policy

import os.path as osp

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

# ==========================================================================
# OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
# HovorkaIntervalRandom starts at a random value
# ==========================================================================

def run_task(*_):

    # ===================
    # Defining the policy
    # ===================
    env = normalize(GymEnv('HovorkaInterval-v0'))
    # env.wrapped_env.env.env.env.reward_flag = 'absolute'
    env.wrapped_env.env.env.reward_flag = 'absolute'
    ## Setting the neural network architecture
    # hidden_sizes=(8,)
    # hidden_sizes=(32, 32)
    # hidden_sizes=(100, 50, 25)

    learn_std = True
    init_std = 1

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        learn_std=learn_std,
        init_std=init_std
    )

    # =======================
    # Defining the algorithm
    # =======================
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    batch_size = 5000
    n_itr = 3
    gamma = .99
    step_size = 0.01
    max_path_length = 96

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=gamma,
        step_size=step_size
    )
    algo.train()

    return algo

# hidden_arc = [str(i) for i in hidden_sizes]
# hidden_arc = '_'.join(hidden_arc)i

# data_dir = 'Reinforce_batchSize_{}_nIters_{}_stepSize_{}_gamma_{}_initStd_{}{}_policyPar_{}_reward_{}'\
        # .format(batch_size, n_itr, step_size,''.join(str(gamma).split('.')), init_std, learn_std, hidden_arc, reward_fun)

data_dir = 'VPG_default'
PROJECT_PATH = '/Users/jonas/Dropbox/results/miguel_experiments/'
log_dir = PROJECT_PATH + data_dir
log_dir='./'

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
    seed=1,
    mode="local",
    plot=False,
    use_cloudpickle=False,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)

## Testing the policy
filename = log_dir + '/params.pkl'
figure_filename = data_dir + '.png'

#render_and_plot_policy(filename, figure_filename)
