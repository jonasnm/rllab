from rllab.algos.tnpg import TNPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

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
    env = normalize(GymEnv('HovorkaInterval-v0'))
    # env.wrapped_env.env.env.env.reward_flag = 'absolute'
    env.wrapped_env.env.env.reward_flag = 'gaussian'


    baseline = LinearFeatureBaseline(env_spec=env.spec)

    learn_std = True
    init_std=1

    # hidden_sizes=(8,)
    hidden_sizes=(32, 32)
    # hidden_sizes=(100, 50, 25)

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
    gamma = .9
    step_size = 0.01

    algo = TNPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        n_itr=n_itr,
        discount=gamma,
        step_size=step_size
    )
    algo.train()

    return algo


# log_dir = '~/Dropbox/results/jonas_experiments/no_stub/'
DROPBOX_DIR = '/home/jonas/Dropbox/results/jonas_experiments/'
# log_dir = DROPBOX_DIR + 'tnpg/gaussian/5000/8'
log_dir = DROPBOX_DIR + 'tnpg/gaussian/5000/32_32'
# log_dir = DROPBOX_DIR + 'tnpg/gaussian/5000/100_50_25'
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
    # exp_prefix="TNPG_" + '32_32_',
    # exp_prefix=data_dir
    plot=False
)


