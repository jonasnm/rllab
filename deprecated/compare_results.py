import joblib
from rllab.sampler.utils import rollout
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

# def render_and_plot_policy(algorithm, filename, figure_filename, title=None):
# def render_and_plot_policy(filename, figure_filename, title=None, plot_average_return=False):
filename = '/home/jonas/Dropbox/results/attd_final_attempt/TRPO/HovorkaBinary-v0/params.pkl'
figure_filename = 'binary3.png'
title = 'TRPO'


data = joblib.load(filename)
policy = data['policy']
env = data['env']
env.wrapped_env.env.env.env.reset_basal_manually = 6.0# algo = data['algo']

path = rollout(env, policy, max_path_length=48,
               animated=True, speedup=1, always_return_paths=True)


bg_history = [path['observations'][i][0:30] for i in range(len(path['observations']))]
bg_history = np.concatenate(bg_history).ravel()

txt= 'A test scenario with three meals and boluses based on erroneous carbohydrate estimates.  Left panel: Basal rate controlled by the trained TRPO algorithm. Right panel: Open loop with a fixed basal rate'
fig = plt.figure(figsize=(12.0, 10.0))
fig.text(.5, .01, txt, ha='center')
plt.ion()

plt.subplot(1, 2, 1)
plt.plot(bg_history)
plt.axhline(y=108, color='r')
plt.axhline(y=180, color='yellow')
plt.axhspan(80, 130, alpha=0.1, color='green')
plt.ylim((60,300))
plt.ylabel('mg/dL')
plt.xlabel('Minutes')
plt.title('Closed loop')

# Making sure the episode is finished!
for i in range(48):
    env.step(np.array([-0.74]))

plt.subplot(1, 2, 2)
env.reset()
for i in range(48):
    env.step(np.array([-0.74]))
plt.plot(env.wrapped_env.env.env.env.bg_history)
plt.axhline(y=108, color='r')
plt.axhline(y=180, color='yellow')
plt.axhspan(80, 130, alpha=0.1, color='green')
plt.ylim((60, 300))
plt.ylabel('mg/dL')
plt.xlabel('Minutes')
plt.title('Open loop')

    # # plt.plot(insulin_history)
    # plt.step(basal_index, basal_rate)
    # plt.ylim((-0.5, 55))
    # plt.ylabel('mU/L')
    # plt.xlabel('Minutes')
    # plt.title('Insulin basal rate')

plt.suptitle(title)

plt.show()
plt.savefig(str(figure_filename))

