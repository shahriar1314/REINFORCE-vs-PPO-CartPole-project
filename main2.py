import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Shared Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #Ensure numerical stability
        return torch.softmax(x - torch.max(x), dim=-1).clamp(min=1e-6)

# REINFORCE Algorithm
class REINFORCE:
    def __init__(self, env):
        self.env = env
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.003)

    def train(self, episodes=100):
        rewards_per_episode = []
        for episode in range(episodes):
            log_probs = []
            rewards = []
            state = self.env.reset()[0]
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_probs = self.policy(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[action])

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)

            returns = self.compute_returns(rewards)
            #loss = -sum(lp * ret for lp, ret in zip(log_probs, returns))
            #advantages = returns - returns.mean()
            #loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages))

            # Calculate advantages (returns - returns.mean())
            advantages = returns - returns.mean()

            # Calculate the loss with advantages
            advantage_loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages))

            # Calculate the entropy term
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))

            # Combine the advantage-based loss and entropy
            total_loss = advantage_loss + 0.005 * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            rewards_per_episode.append(sum(rewards))

            if episode % 25 == 0:
                print(f"REINFORCE - Episode {episode}, Total Reward: {sum(rewards)}")

        return rewards_per_episode

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

# PPO Algorithm
class PPO:
    def __init__(self, env):
        self.env = env
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.eps_clip = 0.2

    def train(self, episodes=100, update_steps=5):
        rewards_per_episode = []
        for episode in range(episodes):
            log_probs = []
            rewards = []
            states = []
            actions = []
            state = self.env.reset()[0]
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_probs = self.policy(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

                log_prob = torch.log(action_probs[action])

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)

            returns = self.compute_returns(rewards)
            rewards_per_episode.append(sum(rewards))

            for _ in range(update_steps):
                new_log_probs = torch.stack([torch.log(self.policy(s)[a]) for s, a in zip(states, actions)])
                ratios = torch.exp(new_log_probs - torch.stack(log_probs).detach())

                advantages = torch.tensor(returns) - torch.mean(torch.tensor(returns))

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Add gradient clipping
                self.optimizer.step()
            if episode % 25 == 0:
                print(f"PPO - Episode {episode}, Total Reward: {sum(rewards)}")

        return rewards_per_episode

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

# Comparison
def compare_algorithms():
    env = gym.make('CartPole-v1')

    reinforce_agent = REINFORCE(env)
    reinforce_rewards = reinforce_agent.train(episodes=1000)

    ppo_agent = PPO(env)
    ppo_rewards = ppo_agent.train(episodes=500)

    plt.plot(reinforce_rewards, label='REINFORCE')
    plt.plot(ppo_rewards, label='PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Performance Comparison: REINFORCE vs PPO on CartPole')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_algorithms()
