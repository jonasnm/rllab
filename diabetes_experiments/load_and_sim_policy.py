import joblib
from rllab.sampler.utils import rollout
import matplotlib.pyplot as plt
import numpy as np
import argparse

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

# def render_and_plot_policy(algorithm, filename, figure_filename, title=None):
def render_and_plot_policy(filename, figure_filename, title=None, plot_average_return=False):

    data = joblib.load(filename)
    policy = data['policy']
    env = data['env']
    # algo = data['algo']

    path = rollout(env, policy, max_path_length=96,
                       animated=True, speedup=1, always_return_paths=True)


    bg_history = [path['observations'][i][0:30] for i in range(len(path['observations']))]
    bg_history = np.concatenate(bg_history).ravel()

    insulin_history = [path['observations'][i][30:] for i in range(len(path['observations']))]
    insulin_history = np.concatenate(insulin_history).ravel()

    plt.figure(figsize=(12.0, 10.0))
    plt.ion()
    plt.subplot(2, 2, 1)
    plt.plot(bg_history)
    plt.axhline(y=108, color='r')
    plt.ylim((60,300))
    plt.ylabel('mg/dL')
    plt.xlabel('Minutes')
    plt.title('Blood glucose level')

    plt.subplot(2, 2, 2)
    plt.plot(insulin_history)
    plt.ylim((-0.5, 20))
    plt.ylabel('mU/L')
    plt.xlabel('Minutes')
    plt.title('Insulin in the body')

    plt.subplot(2, 2, 3)
    plt.plot(path['rewards'])
    plt.ylim((-1.5, 1.5))
    plt.xlabel('Number of actions')
    plt.title('Reward')

    plt.subplot(2, 2, 4)
    plt.plot(path['actions'])
    plt.ylim((-1.5, 1.5))
    plt.ylabel('mU/min (normalized)')
    plt.xlabel('Number of actions')
    plt.title('Actions')


    # hidden_sizes = str(policy.get_param_shapes())

    # =================================
    # Modified with simpler title !!
    # =================================
    plt.suptitle(title)

    # if not title:
         # plt.suptitle(algorithm + 'algorithm, batch size: {}, # iters: {} and gamma: {}. \n NN policy architecture: {}, \n reward fn: {}'.
                 # format(algo.batch_size, algo.n_itr, algo.discount, hidden_sizes, env.wrapped_env.env.env.env.reward_flag))
    # else:

         # plt.suptitle(algorithm + 'algorithm, batch size: {}, # iters: {}, step size: {}, gamma: {}, init stdev: {}{} \n NN policy architecture: {}, \n reward fn: {}'
                      # .format(title['batch_size'], title['n_itr'], title['step_size'],''.join(str(title['gamma']).split('.')), title['init_std'], title['learn_std'], title['hidden_arc'], title['reward_fun']))
    # # suptitle('Reinforce')
    plt.show()
    plt.savefig(str(figure_filename))

if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='The parameter file of the policy to be loaded.')
    parser.add_argument('-ff','--figure_filename', help='filename if figure should be saved')
    parser.add_argument('-t','--title', help='Title of figure', default=None)

    args = parser.parse_args()
    render_and_plot_policy(args.filename, args.figure_filename, args.title)
