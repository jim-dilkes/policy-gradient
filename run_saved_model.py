import gymnasium as gym
import torch
from policy_model import PyTorchWrapper, MultiLayerModel

env_name = "LunarLander-v2"
# env_name = "CartPole-v1"

env = PyTorchWrapper(gym.make(env_name, render_mode="human"))
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

model_layers = [128, 64, 64]

model = MultiLayerModel(n_inputs, n_actions, model_layers)
# Load pytorch model
state_dict = torch.load(f"{env_name}.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Load openai lunar lander env
# env = PyTorchWrapper(gym.make("CartPole-v1", render_mode="human"))
n_steps = 1000
observation, info = env.reset()
print("Running")
for i in range(n_steps):
    env.render()
    action_dist = model(observation)
    action = torch.argmax(action_dist)
    observation, reward, terminated, truncated, info = env.step(action.item())

    if terminated or truncated:
        break
print(f"Done after {i} steps")

env.close()




