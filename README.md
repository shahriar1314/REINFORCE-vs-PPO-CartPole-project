# Reinforcement Learning: REINFORCE vs PPO on CartPole

This project compares the performance of two popular reinforcement learning algorithms, **REINFORCE** and **Proximal Policy Optimization (PPO)**, on the **CartPole-v1** environment from OpenAI Gym.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction
Reinforcement learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. This project implements and compares **REINFORCE** and **PPO** algorithms to solve the CartPole balancing task.

### Algorithms Used:
- **REINFORCE** (Policy Gradient): A Monte Carlo-based policy gradient method that updates policies based on episodic rewards.
- **PPO** (Proximal Policy Optimization): An advanced policy optimization algorithm that improves stability and sample efficiency.

## Project Structure
```
├── data/                   # Folder for storing training logs and results
├── models/                 # Folder for saving trained models
├── src/                    # Source code for training and evaluation
│   ├── train_reinforce.py  # Training script for REINFORCE
│   ├── train_ppo.py        # Training script for PPO
│   ├── plot_results.py     # Visualization script for comparing algorithms
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## Installation
To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/shahriar1314/reinforcement-learning-cartpole.git
   cd reinforcement-learning-cartpole
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Train and evaluate the models using the following commands:

### Train REINFORCE:
```bash
python src/train_reinforce.py
```

### Train PPO:
```bash
python src/train_ppo.py
```

### Visualize Results:
```bash
python src/plot_results.py
```

## Results
The figure below shows the total reward per episode for both algorithms:

<img src="https://github.com/user-attachments/assets/ed46fd6d-55aa-4f88-ae71-f551aaaddafd" alt="Performance Comparison" width="500">




### Observations:
- **PPO** learns faster but has higher variance.
- **REINFORCE** is slower but stabilizes over time.
- Long-term training results in both algorithms achieving maximum rewards.

## References
- OpenAI Gym: https://gym.openai.com/
- PPO Paper: https://arxiv.org/abs/1707.06347
- Policy Gradient Methods: https://spinningup.openai.com/en/latest/

---
Developed by **Shahriar Hassan**. Contributions and suggestions are welcome!

