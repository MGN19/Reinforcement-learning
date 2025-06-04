import gymnasium as gym
import numpy as np

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
