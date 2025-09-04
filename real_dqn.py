'''
The instability of training in carpole_naive_dqn.py is correlation in two forms:
(P1) between observations
(P2) between policy and observations (chasing one's own tail)
most reinforcement learning problem observations are time-series by nature.
correlation is inherent in time-series problem.
while correlation is conducive for convergence in learning, it carries valuable causal information.
so we have to deal with correlation in two manners:
(A) we keep the good part (causal information) by stacking sequential observation, which is referred as preprocessing in
the paper.
(B) we eliminate the bad part (correlation) by
(B1) experience replay to address (P1)
(B2) target network to address (P2)
'''


import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# ale_py.register_v5_envs()

print(gym.__version__)
# unlike "Breakout-v4" that randomly skips frame, "BreakoutNoFrameskip-v4" leaves you on how to handle frame skipping
env = gym.make('BreakoutNoFrameskip-v4')
print(type(env))
env = AtariPreprocessing(
    env,
    noop_max=30,    # do nothing for a random number of seconds [0,30] at the episode start
    frame_skip=4,    # repeats chosen action for 4 consecutive frames to reduce computation load
    screen_size=84,    # Resize to 84x84
    terminal_on_life_loss=True,    # End episode on life loss
    grayscale_obs=True,    # Convert to grayscale to alleviate compute resource
    scale_obs=True)    # Scale pixel values to [0,1]
env = FrameStackObservation(env, stack_size=4)    # Stack 4 consecutive frames to incorporate temporal information
print(type(env))

