from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from pylab import plot, figure, show, title, ion, \
     subplot, suptitle

env = normalize(GymEnv('HovorkaDiabetes-v0'))
# env = (GymEnv('HovorkaDiabetes-v0'))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32),
    # hidden_sizes=(100, 50, 25),
    learn_std=True,
    # init_std=.001
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    # baseline=baseline,
    baseline=baseline,
    batch_size=4001,
    max_path_length=env.horizon,
    n_itr=100,
    # discount=0.80,
    discount=.9,
    step_size=0.01
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)
algo.train()


## Testing the policy

reward = []
actions = []

s = env.reset()

# done = False

# Testing the algorithm
# while not done:
for i in range(1000):

    # Get action recommended by policy
    action = algo.policy.get_action(s)

    # Take the action
    s, r, done, info = env.step(action[0])

    # Check reward and what action was taken
    reward.append(r)
    actions.append(action[0])


# Plotting
figure()
ion()
subplot(2, 2, 1)
plot(env.wrapped_env.env.env.bg_history)
title('Result of running the algorithm')

subplot(2, 2, 3)
plot(reward)
title('Reward')

subplot(2, 2, 4)
plot(actions)
title('Actions')

batch_size = str(algo.batch_size)
n_itr = str(algo.n_itr)
gamma = str(algo.discount)

suptitle('TRPO algorithm, batch size: {}, # iters: {} and gamma: {}. \n NN policy architecture: {}'.format(batch_size, n_itr, gamma, hidden_sizes))
# suptitle('Reinforce')
show()

