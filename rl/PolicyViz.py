import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def qtable_directions_map(qtable, directions : dict, map_size : tuple[int,int]):
    """Get the best learned action & map it based on a dictionnary."""
    qtable_val_max = []
    qtable_best_actions = []
    for state in range(map_size[0]*map_size[1]):
        q_val_max = qtable[state].max()
        best_action = qtable[state].argmax()
        qtable_val_max.append(q_val_max)
        qtable_best_actions.append(best_action)
    qtable_directions = np.empty(len(qtable_best_actions), dtype=str)
    for idx, val in enumerate(qtable_best_actions):
        if qtable_val_max[idx] !=0 :
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size)
    return np.array(qtable_val_max).reshape(map_size), qtable_directions


def plot_directions_heatmap(qtable_val_max,qtable_directions, ax = None, algorithm ="", **kwargs):
    """Plot the best_actions on a heatmap for an environment with low number of states"""
    return sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
            ax = ax
        ).set(title=f"{algorithm} Learned Q-values\nArrows represent best action")

def plot_episodes_info(x, rolling_length = 50, ax = None, **kwargs): 
    """
    Plot training/testing characteristics over episodes and assign a rolling average of the data to provide a smoother graph

    Parameters
    ----------
    x : array_like
    rolling_length : int, optional
        the rolling average length with which to smooth over the graph. The default is 50.
    ax : optional
        a matplotlib.pyplot axis on which to plot the graph. The default is None.
    **kwargs : keyword arguments
        any keyword arguments to provide to seaborn's lineplot

    Returns
    -------
    seaborn lineplot

    """
    reward_moving_average = (
        np.convolve(
            np.array(x).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    
    return sns.lineplot(x = range(len(reward_moving_average)), y = reward_moving_average, ax = ax, **kwargs)
  

def plot_steps_and_rewards(rewards, steps, training_error, axs, **kwargs):
    """Plot the steps and rewards."""
    axs[0].set_title("Episode rewards")
    plot_episodes_info(rewards,ax = axs[0], **kwargs)
    axs[1].set_title("Episode lengths")
    plot_episodes_info(steps,ax = axs[1], **kwargs)
    axs[2].set_title("Training Error")
    plot_episodes_info(training_error,ax = axs[2], **kwargs)
    plt.tight_layout()

