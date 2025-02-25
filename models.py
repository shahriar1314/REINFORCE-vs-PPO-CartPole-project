import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc_policy = nn.Linear(128, action_dim)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc_policy(x), dim=-1)
        value = self.fc_value(x)
        return action_probs, value
