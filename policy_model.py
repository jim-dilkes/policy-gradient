import gymnasium as gym
import torch
from torch import nn

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
    def __init__(self, n_env_inputs, n_action_space, n_per_hidden, activation=nn.ReLU, output_activation=nn.Softmax):
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
        actions = self.output(logits)
        return actions
    