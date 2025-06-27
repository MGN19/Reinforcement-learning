# Reinforcement Learning: Lunar Lander & Atari Pong

Implementation and comparison of reinforcement learning algorithms across two distinct environments: Lunar Lander and Atari Pong.

## 🎯 Project Overview

This project explores the application of several Reinforcement Learning (RL) algorithms to solve two challenging environments with different characteristics:

- **Lunar Lander (v3)**: Continuous observation space with discrete actions
- **Atari Pong (v5)**: High-dimensional visual input with discrete actions

The objective was to implement, compare, and evaluate different RL methods to understand their performance, learning efficiency, and adaptability across varied challenges.

## 🚀 Environments

### Lunar Lander (v3)
- **Observation Space**: Continuous (8 variables: position, velocity, angle, leg contact)
- **Action Space**: Discrete (4 actions: NOOP, Fire Left Engine, Fire Main Engine, Fire Right Engine)
- **Challenge**: Control dynamics and landing precision

### Atari Pong (v5)
- **Observation Space**: High-dimensional visual input (210×160×3 RGB frames)
- **Action Space**: Discrete (6 actions: NOOP, FIRE, UP, DOWN, UPRIGHT, DOWNRIGHT)
- **Challenge**: Visual processing and temporal decision-making

## 🧠 Algorithms Implemented

### Lunar Lander
- **Deep Q-Network (DQN)** - Stable Baselines3
- **Advantage Actor-Critic (A2C)** - Stable Baselines3  
- **Proximal Policy Optimization (PPO)** - Stable Baselines3
- **Custom DQN** - PyTorch implementation from scratch

### Atari Pong
- **Deep Q-Network (DQN)** - Stable Baselines3
- **Quantile Regression DQN (QRDQN)** - Stable Baselines3
- **Advantage Actor-Critic (A2C)** - Stable Baselines3
- **Proximal Policy Optimization (PPO)** - Stable Baselines3 (with Optuna optimization)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MGN19/Reinforcement-Learning.git
```

### 2. Install Dependencies
For Lunar

```bash
pip install -r requirements.txt
```

For Pong

```bash
pip install -r requirements_pong.txt
```

### 3. Run Lunar Lander Experiments
```bash
# Train baseline
Lunar Lander.ipynb
```

### 4. Run Pong Experiments
```bash
# Train baseline
Pong.ipynb

# Train with Optuna
Pong_PPO_Optuna.ipynb
```

## 📊 Results

### Lunar Lander
- **Best Algorithm**: Custom DQN (Mean Reward: 246.31 ± 44.17)
- **Training Time**: ~7 minutes (Custom DQN) vs ~16 minutes (Regular DQN)
- **Key Finding**: Simplified custom implementation outperformed complex pre-built frameworks

### Atari Pong
- **Best Algorithm**: PPO (after hyperparameter optimization)
- **Key Finding**: Policy-gradient methods handle high-dimensional visual input better than value-based methods
- **Performance**: Moving average episode return between -12 and -17 after optimization - not able to solve

## 🔍 Key Insights

1. **Algorithm-Environment Compatibility**: PPO was worst in Lunar Lander but best in Pong.

2. **Custom vs Pre-built**: The custom DQN implementation demonstrated that tailored, simplified models can sometimes outperform complex frameworks.

3. **Visual Processing**: Policy-gradient methods (PPO) showed superior performance in high-dimensional visual environments compared to value-based methods.

4. **Hyperparameter Optimization**: Optuna-based tuning slightly improved PPO's performance in Pong.

## 📈 Preprocessing (Pong)

Frame preprocessing pipeline for Atari Pong:
- **Grayscale Conversion**: RGB (210×160×3) → Grayscale
- **Resizing**: 210×160 → 84×84 pixels
- **Frame Stacking**: Stack 4 consecutive frames for temporal information
- **Dimensionality Reduction**: 100,800 → 28,224 input features

## 📚 References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML..
