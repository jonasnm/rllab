from rllab.sampler.utils import rollout
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

# def render_and_plot_policy(algorithm, filename, figure_filename, title=None):
def plot_training_progress(filename, figure_filename=None, title=None):

    plt.figure(figsize=(12.0, 10.0))

    # Load data
    data = pd.read_csv(filename)

    avg_ret = np.array(data["AverageReturn"])
    std_dev_ret = np.array(data["StdReturn"])
    smoothed_rewards = pd.Series(avg_ret).rolling(5, min_periods=5).mean()

    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, color='orange')

    plt.fill_between(range(len(smoothed_rewards)), smoothed_rewards + std_dev_ret, smoothed_rewards - std_dev_ret, alpha=0.3, edgecolor='orange', facecolor='orange')

    if title is not None:
        plt.title(title)
    else:
        plt.title('Average return during training')

    plt.show()

    plt.savefig(str(figure_filename))

if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='The parameter file of the training progress.')
    parser.add_argument('-ff','--figure_filename', help='filename if figure should be saved')
    parser.add_argument('-t','--title', help='Title of figure', default=None)

    args = parser.parse_args()
    plot_training_progress(args.filename, args.figure_filename, args.title)
