import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from utils import plot_learning_curve


'''
We separately define the network and the agent for easy management in the future. For example, we can swap different
network architectures without modification on agent. We can also independently unit test network and agent.
'''
class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # a callable object, which can be invoked like a regular function
        # a regular function is stateless; a callable object is stateful (can store configuration and data between calls)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0, esp_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = esp_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        # epsilon greedy action selection
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
        # predicted Q-value for the actually taken action in current state basis current model
        q_pred = self.Q.forward(states)[actions]
        # the best Q-value among all possible actions from the next state basis current model
        q_next = self.Q.forward(states_).max()
        q_target = reward + self.gamma * q_next
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []
    agent = Agent(
        input_dims = env.observation_space.shape,
        n_actions = env.action_space.n,
        lr = 0.0001
    )
    for i in range(n_games):
        score = 0
        done = False
        obs, _ = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i}: score {score}, average score {avg_score}, epsilon {agent.epsilon}')

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)