import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

def test_and_plot(env, algo):
    ''' Testing the policy, plotting the result
    and saving the figure
    '''
    ## Testing the policy
    from pylab import plot, figure, show, title, ion, \
         subplot, suptitle

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


    suptitle('Reinforce algorithm, batch size: {}, # iters: {} and gamma: {}. \n NN policy architecture: {}, reward fn: {}'.format(batch_size, n_itr, gamma, hidden_sizes, reward_fun))
    # suptitle('Reinforce')
    show()


def bgplot(blood_glucose_level, block=False):
    """
    Plotting the blood glucose curve
    """
    bg_low = 70 * np.ones(len(blood_glucose_level))
    bg_high = 170 * np.ones(len(blood_glucose_level))
    # n_days = int(len(blood_glucose_level)/1440)

    l = int(len(blood_glucose_level))

    fig, ax = plt.subplots()
    ax.plot(blood_glucose_level)
    # ax.plot(70*np.ones(l), color='red')
    # ax.plot(170*np.ones(l), color='red')
    # ax.fill_between(range(0, l), bg_low, bg_high, alpha=.15)
    ax.fill_between(range(0, l), bg_low, bg_high, alpha=.20,
                    facecolor=sns.xkcd_rgb["light orange"])
    plt.title('Blood glucose curve')
    plt.ylabel('[mg/dl]')
    plt.xlabel('Minutes')
    plt.axis([0, l, 0, 230])
    plt.ion()
    plt.show(block=block)


def bgplot_save(blood_glucose_level, filename, block=False):
    """
    Plotting the blood glucose curve
    """
    bg_low = 70 * np.ones(len(blood_glucose_level))
    bg_high = 170 * np.ones(len(blood_glucose_level))
    # n_days = int(len(blood_glucose_level)/1440)

    l = int(len(blood_glucose_level))

    fig, ax = plt.subplots()
    ax.plot(blood_glucose_level)
    # ax.plot(70*np.ones(l), color='red')
    # ax.plot(170*np.ones(l), color='red')
    # ax.fill_between(range(0, l), bg_low, bg_high, alpha=.15)
    ax.fill_between(range(0, l), bg_low, bg_high, alpha=.20,
                    facecolor=sns.xkcd_rgb["light orange"])
#    plt.title('Blood glucose curve')
    plt.ylabel('[mg/dl]')
    plt.xlabel('Minutes')
    plt.axis([0, l, 0, 230])
    plt.ion()
    plt.show(block=block)

    fig.savefig(filename, bbox_inches='tight')
