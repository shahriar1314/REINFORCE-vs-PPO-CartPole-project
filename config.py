import torch 

REINFORCE_CONFIG = {
    'num_episodes': 100,
    'lr': 0.001,
    'gamma': 0.99
}

PPO_CONFIG = {
    'num_episodes': 100,
    'lr': 0.0003,
    'gamma': 0.99,
    'gae_lambda': 0.95
}