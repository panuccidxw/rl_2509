import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

ale_py.register_v5_envs()

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def make_env(env_name):
    env = gymnasium.make(env_name, render_mode = "rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,  # do nothing for a random number of seconds [0,30] at the episode start
        frame_skip=4,  # repeats chosen action for 4 consecutive frames to reduce computation load
        screen_size=84,  # Resize to 84x84
        terminal_on_life_loss=True,  # End episode on life loss
        grayscale_obs=True,  # Convert to grayscale to alleviate compute resource
        scale_obs=True)  # Scale pixel values to [0,1]
    env = FrameStackObservation(env, stack_size=4)  # Stack 4 consecutive frames to incorporate temporal information

    return env