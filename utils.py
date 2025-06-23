import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random


# Random Agent Runner
def run_agent(
    env_name="LunarLander-v3",
    episodes=5,
    render=True,
    seed=None,
    log_fn=None,
    agent_fn=None
):


    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)

    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if agent_fn:
                action = agent_fn(obs)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        if log_fn:
            log_fn(episode + 1, total_reward)
        else:
            print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")

    env.close()
    return episode_rewards

# Logging function 
def my_logger(ep, reward):
    print(f"[LOG] Episode {ep} - Reward: {reward:.2f}")


# Rule-based agent for LunarLander
def rule_based_action(obs):
    x, y, x_dot, y_dot, angle, angle_vel, left_leg, right_leg = obs

    # If on the ground with both legs, do nothing
    if left_leg and right_leg:
        return 0

    # If descending too fast, fire main engine
    if y_dot < -0.5:
        return 2

    # Try to stabilize orientation
    if angle > 0.1:
        return 1  # fire left engine to rotate right
    elif angle < -0.1:
        return 3  # fire right engine to rotate left

    # Otherwise, occasionally fire main engine to keep from falling
    if y < 0.5:
        return 2

    return 0  # default to doing nothing

# Episodes for Stable Baselines3 model
def run_episodes(model, env, n_episodes=10):
    episode_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()  
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
    return episode_rewards

## Torch DQN Implementation

# Define the Q-network (Deep Q-Network - DQN)
class QNet(nn.Module):

    # Fully connected layers: input → hidden → hidden → output
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(), # Activation function
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    # Forward pass: compute Q-values for a given batch of states
    def forward(self, x):
        return self.fc(x)

    # Predict method: returns the best action for a given observation
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.fc[0].weight.device)
            action = self.forward(obs_tensor).argmax().item()
        return action, None

# Action selection with ε-greedy
def select_action(model, state, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        action, _ = model.predict(state)
        return action
    
# Model training function
def train_dqn(model, env, optimizer, device, episodes, gamma, 
              epsilon_start, epsilon_end, epsilon_decay, max_steps=1000):
    reward_history = []
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            # Select action (ε-greedy)
            action = select_action(model, state, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Compute target Q-value
            with torch.no_grad():
                next_q = model(torch.FloatTensor(next_state).unsqueeze(0).to(device))
                target = reward + gamma * next_q.max().item() * (1 - done)

            # Current Q-value
            q_vals = model(torch.FloatTensor(state).unsqueeze(0).to(device))
            q_val = q_vals[0, action]

            # Loss and backpropagation
            loss = (q_val - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

        reward_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode}: Reward = {total_reward:.1f}, Epsilon = {epsilon:.3f}")

    return model, reward_history
