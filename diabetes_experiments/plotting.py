import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn for better plotting!')

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
