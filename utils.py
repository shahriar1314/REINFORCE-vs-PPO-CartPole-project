import numpy as np

def discount_rewards(rewards, gamma):
    discounted = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted

def compute_gae(rewards, values, gamma, gae_lambda):
    advantages = []
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    return returns, advantages