import gymnasium as gym
import torch
from policy_model import PyTorchWrapper, Agent
from functools import partial

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

env_name = "LunarLander-v2"
# env_name = "CartPole-v1"

env = PyTorchWrapper(gym.make(env_name, render_mode="human"), device)
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

model_layers = [128,64]
output_activation = partial(torch.nn.Softmax, dim=-1)
model = Agent(n_inputs, n_actions, model_layers)

# Load pytorch model
state_dict = torch.load(f"{env_name}.pt", map_location=torch.device(device))
model.load_state_dict(state_dict)

# Load openai lunar lander env
# env = PyTorchWrapper(gym.make("CartPole-v1", render_mode="human"))
n_steps = 500
observation, info = env.reset()
print("Running")

cum_reward = 0
for i in range(n_steps):
    env.render()
    action = model.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    cum_reward += reward

    if terminated or truncated:
        break
    
print(f"Done after {i} steps with cumulative reward {cum_reward}")

env.close()




