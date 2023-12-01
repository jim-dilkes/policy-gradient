import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

class PyTorchWrapper(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return torch.from_numpy(observation).float().to(self.device), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = torch.from_numpy(observation).float().to(self.device)
        return observation, reward, terminated, truncated, info


class MultiLayerModel(nn.Module):
    def __init__(self, n_env_inputs, n_action_space, n_per_hidden, output_activation, activation=nn.ReLU):
        super().__init__()
        n_per_hidden = [n_per_hidden] if type(n_per_hidden) != list else n_per_hidden
        layer_counts = [n_env_inputs] + n_per_hidden + [n_action_space]
        
        layers = []
        for i in range(len(layer_counts) - 1):
            layers.append(nn.Linear(layer_counts[i], layer_counts[i + 1]))
            if i < len(layer_counts) - 2:
                layers.append(activation())
        
        self.linear_relu_stack = nn.Sequential(*layers)
        self.output = output_activation() if output_activation is not None else lambda x: x
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.output(logits)


class Agent():
    def __init__(self, n_env_inputs, n_action_space, n_per_hidden, batch_size=1,
                policy_optimizer=torch.optim.Adam,
                policy_lr=1e-3):
        self.policy_model = MultiLayerModel(n_env_inputs, n_action_space, n_per_hidden, partial(torch.nn.Softmax, dim=-1), activation=nn.ReLU)
        
        self.rewards = []
        self.log_probs = []

        self.policy_optimizer = policy_optimizer(self.policy_model.parameters(), lr=policy_lr)

        self.batch_size = batch_size
        self.batch_count = 0

    def set_policy_learning_rate(self, lr):
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, path):
        if path[-3:] != ".pt":
            path += ".pt"
        torch.save(self.policy_model.state_dict(), path)

    def load_state_dict(self, state_dict):
        self.policy_model.load_state_dict(state_dict)

    def to(self, device):
        self.policy_model.to(device)
        return self

    def reset(self):
        self.rewards = []
        self.log_probs = []
        self.batch_count = 0

        self.policy_optimizer.zero_grad()

    def select_action(self, state):
        probs = self.policy_model.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def calculate_returns(self, gamma=1, normalize=True):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def update_parameters(self, gamma=1):
        returns = self.calculate_returns(gamma, normalize=True)
        loss = -torch.stack(self.log_probs).squeeze() * returns
        loss = sum(loss)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        del self.rewards[:]
        del self.log_probs[:]


class BaselineAgent(Agent):
    '''Agent with a value model to estimate the value of a state.'''
    def __init__(self, n_env_inputs, n_action_space, n_per_hidden, batch_size=1,
                 policy_optimizer=torch.optim.Adam, policy_lr=1e-3,
                 value_optimizer=torch.optim.Adam, value_lr=1e-2):
        super().__init__(n_env_inputs, n_action_space, n_per_hidden, batch_size, policy_optimizer, policy_lr)
        self.value_model = MultiLayerModel(n_env_inputs, 1, n_per_hidden, None, activation=nn.ReLU)
        self.value_optimizer = value_optimizer(self.value_model.parameters(), lr=value_lr)

        self.states = []

    def select_action(self, state):
        action = super().select_action(state)
        self.states.append(state)
        return action

    def to(self, device):
        self.value_model.to(device)
        super().to(device)
        return self

    def reset(self):
        super().reset()
        self.states = []
        self.value_optimizer.zero_grad()

    def set_value_learning_rate(self, lr):
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr

    def update_parameters(self, gamma=1):

        # Calculate advantages
        returns = self.calculate_returns(gamma, normalize=False)
        value_estimates = self.value_model(torch.stack(self.states)).squeeze()
        advantages = (returns - value_estimates).detach()

        # Update value model
        value_loss = -value_estimates * advantages
        value_loss = value_loss.sum()
        value_loss.backward()
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()

        # Update policy model
        policy_loss = -torch.stack(self.log_probs).squeeze() * advantages
        policy_loss = sum(policy_loss)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        # Clear memory
        del self.rewards[:]
        del self.log_probs[:]
        del self.states[:]