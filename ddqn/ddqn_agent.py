import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.online_net = DeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name=self.env_name+'_'+self.algo+'_q_eval',
                                       chkpt_dir=self.chkpt_dir)
        self.target_net = DeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name=self.env_name+'_'+self.algo+'_q_next',
                                       chkpt_dir=self.chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.online_net.device)
        rewards = T.tensor(reward).to(self.online_net.device)
        dones = T.tensor(done).to(self.online_net.device)
        actions = T.tensor(action).to(self.online_net.device)
        states_ = T.tensor(new_state).to(self.online_net.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.online_net.device)
            actions = self.online_net.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.online_net.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        # see ddqn.drawio for diagram

        q_pred = self.online_net.forward(states)[indices, actions]
        q_next = self.target_net.forward(states_)
        q_eval = self.online_net.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.online_net.loss(q_target, q_pred).to(self.online_net.device)
        loss.backward()

        self.online_net.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.online_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.online_net.load_checkpoint()
        self.target_net.load_checkpoint()