from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite, stub

from plotting import test_and_plot

import os.path as osp

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

# Experiment parameters


# Running the experiment in "stub" mode -- for use with the
# run_experiment_lite function...
stub(globals())

# OpenAI diabetes envs
env_name ='HovorkaInterval-v0'
# env_name = 'HovorkaIntervalRandom-v0'

env = normalize(GymEnv(env_name))


# ==============================
# Changing the reward function
# ==============================

# env.wrapped_env.env.env.reward_flag = 'gaussian'

# =========================================
# Setting the neural network architecture
# =========================================

# hidden_sizes=(8,)
# hidden_sizes=(32, 32)
hidden_sizes=(100, 50, 25)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=hidden_sizes,
    learn_std=True,
    init_std=.01
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

batch_size = 4000
n_itr = 200
gamma = .99
step_size = 0.05

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=batch_size,
    n_itr=n_itr,
    discount=gamma,
    step_size=step_size
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

# String manipulation
reward_fun = 'gaussian_with_insulin'
hidden_arc = [str(i) for i in hidden_sizes]
hidden_arc = '_'.join(hidden_arc)
data_dir = 'Reinforce_batch_size_{}_n_iters_{}_gamma_{}_nnpar_{}_reward_{}'\
        .format(batch_size, n_itr, gamma, hidden_arc, reward_fun)

# data_dir = 'test'

# Running and saving the experiment
run_experiment_lite(
    algo.train(),
    # log_dir=data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="Reinforce_" + env_name,
    seed=1,
    mode="local",
    plot=False,
    # terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)


## Testing the policy
# test_and_plot(env, algo)
