from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from pylab import plot, figure, show, title, ion, \
     subplot, suptitle

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

# import gym


env = normalize(GymEnv('HovorkaInterval-v0'))

env.wrapped_env.env.env.reward_flag = 'gaussian'

# hidden_sizes=(8,)
# hidden_sizes=(32, 32)
hidden_sizes=(100, 50, 25)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=hidden_sizes,
    learn_std=True,
    init_std=.5
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2000,
    n_itr=200,
    # discount=0.80,
    discount=.99,
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
for i in range(96):

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

suptitle('Reinforce algorithm, batch size: {}, # iters: {} and gamma: {}. \n NN policy architecture: {}'.format(batch_size, n_itr, gamma, hidden_sizes))
# suptitle('Reinforce')
show()

