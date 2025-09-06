
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

One of DQN's historical impact is that it is agnostic across different environments (dozens of Atari games).
Before DQN (2015), most game AI was hand-crafted.

dqn_main.py
├── dqn_agent.py
│   ├── deep_q_network.py
│   └── replay_memory.py
└── utils.py
'''

import numpy as np
import time
import os
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gymnasium.wrappers import RecordVideo

if __name__ == '__main__':
    # unlike "Breakout-v4" that skips random number of frames, "BreakoutNoFrameskip-v4" leaves you on how to handle frame skipping
    # other available games: 'PongNoFrameskip-v4', 'CartPole-v1', 'SpaceInvadersNoFrameskip-v4'
    game_name = "SpaceInvadersNoFrameskip-v4"
    game_start_time = time.time()
    env = make_env(game_name)
    best_score = -np.inf
    load_checkpoint = False
    n_games = 501
    """
    batch_size is the trade-off between learning stability vs speed
    batch_size too small: very noisy gradient, unstable learning, fails to learn generally
    batch_size too large: too slow
    don't use very large batch_size just because you have a powerful gpu
    practically, batch size 64 -- 512 is good
    """
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=128, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name=game_name)

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    log_file = 'logs/' + fname + '.txt'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    # RecordVideo() is just a wrapper that preserves original env properties and methods, but adds functionality for video recording
    # Extract game name for video filename (e.g., "PongNoFrameskip-v4" -> "pong") 
    game_prefix = game_name.lower().split('noframeskip')[0].replace('-', '')
    env = RecordVideo(
        env,
        video_folder="video/",
        episode_trigger=lambda episode_id: episode_id % 100 == 0,  # Record every episode
        video_length=0,  # Record entire episode
        name_prefix=f"{game_prefix}-episode"
)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        print(f"running game {i}")
        start_time = time.time()
        done = False
        observation, _ = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        duration = time.time() - start_time

        avg_score = np.mean(scores[-100:])
        log_message = f'episode: {i}, score: {score}, moving average: {avg_score:.1f}, best: {best_score:.1f}, duration: {duration:.1f}'
        print(log_message)
        
        # write to log file
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    # with RecordVideo env wrapper, you need to close the env to finalize the last recording.
    env.close()
    
    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

    game_end_time = time.time()
    total_seconds = int(game_end_time - game_start_time)
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"The entire training took {days} days {hours} hours {minutes} mins {seconds:.1f} seconds")