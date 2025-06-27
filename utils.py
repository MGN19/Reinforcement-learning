import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt


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
def run_episodes_get_rewards(model, env, n_episodes=10):
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

# Evaluate the trained model
def evaluate_model(model, env, n_episodes=100, render=False):
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
            # Use deterministic action (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    env.close()
    avg_reward = sum(rewards) / n_episodes
    std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / n_episodes) ** 0.5

    return rewards, avg_reward, std_reward

## Plots


# Plotting function for reward trend
def plot_reward_trend(episode_logs, window_size=10):
    episode_rewards = [ep["total_reward"] for ep in episode_logs]
    if len(episode_rewards) < window_size:
        window_size = len(episode_rewards)  
    
    rolling_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(range(window_size - 1, len(episode_rewards)), rolling_avg, label=f'{window_size}-Episode Moving Average', color='red')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Trend Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

# Scatter plot of actions taken in an episode
def scatter_plot_actions(actions, episode=0):
    """
    Scatter plot of actions taken over time steps in a single episode.

    actions: List of lists, each inner list contains actions from one episode.
    episode: index of the episode to plot
    """
    episode_actions = actions[episode]

    plt.figure(figsize=(12, 4))
    plt.scatter(range(len(episode_actions)), episode_actions, c=episode_actions, cmap="viridis", s=10)
    plt.xlabel("Time Step")
    plt.ylabel("Action")
    plt.title(f"Actions Scatter Plot in Episode {episode + 1}")
    plt.yticks(np.unique(episode_actions))
    plt.colorbar(label='Action Value')
    plt.grid(True)
    plt.show()


# Plot action usage over time
def plot_action_usage_over_time(episode_logs, chunk_size=50, env=None):
    n_chunks = len(episode_logs) // chunk_size
    action_counts = []
    
    for i in range(n_chunks):
        chunk_actions = [a for ep in episode_logs[i*chunk_size:(i+1)*chunk_size] for a in ep["actions"]]
        counts, _ = np.histogram(chunk_actions, bins=np.arange(env.action_space.n + 1))
        action_counts.append(counts)
    
    action_counts = np.array(action_counts)
    
    plt.figure(figsize=(12,6))
    for action in range(action_counts.shape[1]):
        plt.plot(range(n_chunks), action_counts[:, action], label=f"Action {action}")
    plt.xlabel(f"Episode chunks (size={chunk_size})")
    plt.ylabel("Action count")
    plt.title("Action Usage Over Training")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot episode lengths over time
def plot_episode_lengths(episode_logs):
    lengths = [len(ep["actions"]) for ep in episode_logs]
    plt.figure(figsize=(10,5))
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (timesteps)")
    plt.title("Episode Length over Time")
    plt.grid(True)
    plt.show()