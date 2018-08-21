import joblib
from rllab.sampler.utils import rollout
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

def render_and_plot_policy(filename):

    data = joblib.load(filename)
    policy = data['policy']
    env = data['env']
    algo = data['algo']

    path = rollout(env, policy, max_path_length=96,
                       animated=True, speedup=1, always_return_paths=True)


    bg_history = [path['observations'][i][0:29] for i in range(len(path['observations']))]
    bg_history = np.concatenate(bg_history).ravel()

    plt.figure()
    plt.ion()
    plt.subplot(2, 2, 1)
    plt.plot(bg_history)
    plt.title('Result of running the algorithm')

    plt.subplot(2, 2, 3)
    plt.plot(path['rewards'])
    plt.title('Reward')

    plt.subplot(2, 2, 4)
    plt.plot(path['actions'])
    plt.title('Actions')


    hidden_sizes = str(policy.get_param_shapes())
    plt.suptitle('Reinforce algorithm, batch size: {}, # iters: {} and gamma: {}. \n NN policy architecture: {}, \n reward fn: {}'.
                 format(algo.batch_size, algo.n_itr, algo.discount, hidden_sizes, env.wrapped_env.env.env.env.reward_flag))
    # suptitle('Reinforce')
    plt.show()


